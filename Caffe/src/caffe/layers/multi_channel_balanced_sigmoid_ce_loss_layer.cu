#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_balanced_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiChannelBalancedSigmoidCrossEntropyLossForwardGPU(const int nthreads, const Dtype* input_data, const Dtype* target, Dtype* loss, const int dim, const Dtype* weight) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[(i%dim)]);
  	if (target_value == (int)(i/dim)+1) { /* positive */
      loss[i] = input_data[i] * (1 - (input_data[i] >= 0)) -
       log(1+exp(input_data[i] - 2*input_data[i] * (input_data[i] >= 0)));
			loss[i] *= (1 - weight[target_value-1]);
  	} else { /* negative */
      loss[i] = input_data[i] * (0 - (input_data[i] >= 0)) -
       log(1 + exp(input_data[i] - 2*input_data[i]*(input_data[i] >= 0)));
			loss[i] *= weight[(int)(i/dim)];
  	}
  }
}

template <typename Dtype>
__global__ void MultiChannelBalancedSigmoidCrossEntropyLossBackwardGPU(const int nthreads, Dtype* diff, const Dtype* target, const Dtype* sigmoid_output_data, const int dim, const Dtype *weight) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[(i%dim)]);
  	diff[i] = sigmoid_output_data[i];
  	if (target_value == (int)(i/dim)+1) { /* positive */
  	  diff[i] = diff[i] -1;
      diff[i] *= (1 - weight[target_value-1]);
  	} else { /* negative */
      diff[i] *= weight[(int)(i/dim)];
  	}
  }
}

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* input_data = bottom[0]->gpu_data();  /* pred */
  const Dtype* target = bottom[1]->gpu_data();   /* label */

  const int count = bottom[0]->count();
  const int dim = count / bottom[0]->channels();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  /* calculate weight of negative pixels for different categories */
  const Dtype* target_cpu = bottom[1]->cpu_data();
	Dtype* class_weight = class_weight_.mutable_cpu_data();
  for (int i = 0; i < dim; i++) {
    const int target_value = static_cast<int>(target_cpu[i]);
		if (target_value > 0) 
			class_weight[target_value-1]++;	/* record the number of positive pixels */
  }
	for (int i= 0; i < num_label_; i++) {
		class_weight[i] = 1.0 * class_weight[i] / dim;	/* weight = count / totalcount */
	}

	/* calculate loss for different categories */
	const Dtype* weight_gpu = class_weight_.gpu_data();
  MultiChannelBalancedSigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data, dim, weight_gpu);

	/* sum over all elements */
	Dtype loss = 0;
	caffe_gpu_asum(count, loss_data, &loss);

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	  const Dtype* class_weight = class_weight_.gpu_data();
    int dim = bottom[0]->height()*bottom[0]->width();

    MultiChannelBalancedSigmoidCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, target, sigmoid_output_data, dim, class_weight);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MultiChannelBalancedSigmoidCrossEntropyLossLayer);

}  // namespace caffe
