#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const MCSCELossParameter  mcsce_loss_param = this->layer_param_.mcsce_loss_param();
  num_label_  = mcsce_loss_param.num_label();
  weight_pos_ = 0;
  weight_neg_ = 0;
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), num_label_) <<
	  "Channels must have the same number as num_label";
  CHECK_EQ(bottom[0]->height()*bottom[0]->width(), bottom[1]->count()) <<
      "Multi_Channel_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;

  int dim = bottom[0]->height()*bottom[0]->width();
  /* compute weight_pos_ and weight_neg_ */
  for (int i = 0; i < dim; i++) {
  	if (target[i] != 0) count_pos++;
  	else count_neg++;
  }
  weight_pos_ = count_neg / (count_pos + count_neg);
  weight_neg_ = count_pos / (count_pos + count_neg);

	/* calculate loss over num_label_ channels */
  for (int i = 0; i < num_label_; ++i) { /* loop channels */
  	temp_loss_pos = 0;
  	temp_loss_neg = 0;
    for (int j = 0; j < dim; ++j) {  /* loop every pixels, dim = H*W */
  	  int idx = i*dim+j;
  	  Dtype temp = log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0))); /* avoid redundant computing */
      if (target[j] == (i+1)) { /* positive pixels */
       	temp_loss_pos -= input_data[idx] * (1 - (input_data[idx] >= 0)) - temp;
      } else { /* negative */
        temp_loss_neg -= input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
      }
    } 
    loss_pos += temp_loss_pos * weight_pos_; /* sum over channels */
    loss_neg += temp_loss_neg * weight_neg_;
  }
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
}

template <typename Dtype>
void MultiChannelSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int dim = bottom[0]->height()*bottom[0]->width();

	  /* calculate gradient */
    for (int i = 0; i < num_label_; ++i) { /* loop over channels */
      for (int j = 0; j < dim; ++j) { /* loop over pixels */
	      int idx = i*dim+j;
      	if (target[j] == (i+1)) {
	  	    bottom_diff[idx] = sigmoid_output_data[idx] - 1;
          bottom_diff[idx] *= weight_pos_; /* weight_pos_ was calculated in forward phase */
      	} else {
	  	    bottom_diff[idx] = sigmoid_output_data[idx] - 0;
          bottom_diff[idx] *= weight_neg_;
      	}
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MultiChannelSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MultiChannelSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiChannelSigmoidCrossEntropyLoss);

}  // namespace caffe
