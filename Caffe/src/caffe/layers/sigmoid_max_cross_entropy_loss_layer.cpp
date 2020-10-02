#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_max_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidMaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  count_pos_ = 0;
  neg_index_ = -1;
}

template <typename Dtype>
void SigmoidMaxCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_MAX_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidMaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_neg = 0;

  int dim = bottom[0]->count() / bottom[0]->num();
  for (int j = 0; j < dim; j ++) {
     if (target[j] == 1) {
    	count_pos_ ++;
    	loss_pos -= input_data[j] * (target[j] - (input_data[j] >= 0)) -
            	log(1 + exp(input_data[j] - 2 * input_data[j] * (input_data[j] >= 0)));
    } else if (target[j] == 0) {
    	temp_loss_neg = -input_data[j] * (target[j] - (input_data[j] >= 0))+ 
            	log(1 + exp(input_data[j] - 2 * input_data[j] * (input_data[j] >= 0)));
		if (temp_loss_neg > loss_neg) {
			loss_neg = temp_loss_neg;
			neg_index_ = j;
		} 
     }
  } 
  loss_pos *= 1.0 / (1.0 + count_pos_);
  loss_neg *= count_pos_ / (1.0 + count_pos_);
  std::cout << "neg_index " << neg_index_ << std::endl;

  top[0]->mutable_cpu_data()[0] = (loss_pos  + loss_neg);
}

template <typename Dtype>
void SigmoidMaxCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    int dim = bottom[0]->count() / bottom[0]->num();
    for (int j = 0; j < dim; j ++) {
      if (target[j] == 1) {
        bottom_diff[j] *= 1.0 / (1.0 + count_pos_);
      } else if (j == neg_index_) {
	    bottom_diff[j] *= count_pos_ / (1.0 + count_pos_);
	  } else {
	    bottom_diff[j] *= 0;
	  }
    }

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidMaxCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidMaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidMaxCrossEntropyLoss);

}  // namespace caffe
