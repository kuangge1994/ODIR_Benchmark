#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiChannelSigmoidCrossEntropyLossForwardGPU(const int nthreads, const Dtype* input_data, const Dtype* target, Dtype* loss, const int dim,  const Dtype weight_pos, const Dtype weight_neg) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[(i%dim)]);
	  if (target_value == (int)(i/dim)+1) { /* positive */
      loss[i] = input_data[i] * (1 - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] *
        (input_data[i] >= 0)));
			loss[i] *= weight_pos;
	  } else { /* negative */
      loss[i] = input_data[i] * (0 - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] *
        (input_data[i] >= 0)));
			loss[i] *= weight_neg;
	  }
  }
}

template <typename Dtype>
__global__ void MultiChannelSigmoidCrossEntropyLossBackwardGPU(const int nthreads, Dtype* diff, const Dtype* target, const Dtype* sigmoid_output_data, const int dim, double weight_pos, double weight_neg) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[(i%dim)]);
  	diff[i] = sigmoid_output_data[i];
  	if (target_value == (int)(i/dim)+1) { /* positive */
  	  diff[i] = diff[i] -1;
      diff[i] *= weight_pos;
  	} else { /* negative */
      diff[i] *= weight_neg;
  	}
  }
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* input_data = bottom[0]->gpu_data();  /* pred */
  const Dtype* target = bottom[1]->gpu_data();   /* label */

  const int count = bottom[0]->count();
  const int dim = count / bottom[0]->channels();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype count_pos = 0;
  Dtype count_neg = 0;
  const Dtype* target_cpu = bottom[1]->cpu_data();
  /* calculate beta and (1-beta) */
  for (int i = 0; i < dim; i++) {
	  if (target_cpu[i] != 0) count_pos++;
	  else count_neg++;
  }
  weight_pos_ = 1.0 * count_neg / (count_pos + count_neg);
  weight_neg_ = 1.0 * count_pos / (count_pos + count_neg);

  MultiChannelSigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data, dim, weight_pos_, weight_neg_);

	Dtype loss = 0;
	caffe_gpu_asum(count, loss_data, &loss);

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
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

    int dim = bottom[0]->height()*bottom[0]->width();
		CHECK_GE(weight_pos_, 0) << "error";
		CHECK_GE(weight_neg_, 0) << "error";

    MultiChannelSigmoidCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, target, sigmoid_output_data, dim, (double)weight_pos_, (double)weight_neg_);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MultiChannelSigmoidCrossEntropyLossLayer);

}  // namespace caffe
