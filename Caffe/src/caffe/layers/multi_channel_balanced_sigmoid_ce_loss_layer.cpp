#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_balanced_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const MCBSCELossParameter  mcbsce_loss_param = this->layer_param_.mcbsce_loss_param();
  num_label_ = mcbsce_loss_param.num_label();
}

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), num_label_) <<
	  "Channels must have the same number as num_label";
  CHECK_EQ(bottom[0]->height()*bottom[0]->width(), bottom[1]->count()) <<
      "Multi_Channel_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
	class_weight_.Reshape(num_label_,1,1,1); /* store weight for different channels */
}

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
	Dtype* class_weight = class_weight_.mutable_cpu_data();
	caffe_set(class_weight_.count(), Dtype(0), class_weight);
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;

  int dim = bottom[0]->height()*bottom[0]->width();
  for (int i = 0; i < num_label_; ++i) { /* loop over channels */
	  temp_loss_pos = 0;
	  temp_loss_neg = 0;
    for (int j = 0; j < dim; ++j) { /* loop over elements, dim = H*W */
		  int idx = i*dim+j;
		  Dtype temp = log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      if (target[j] == (i+1)) { /* target_j equals channel number, marked as positive */
		  	class_weight[i]++;
        temp_loss_pos -= input_data[idx] * (1 - (input_data[idx] >= 0)) - temp;
      } else { /* negative */
     	  temp_loss_neg -= input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
    	}
    } 
		class_weight[i] /= dim; /* class_weight: stores the weight for negative pixels */
    loss_pos += temp_loss_pos * (1 - class_weight[i]);
    loss_neg += temp_loss_neg * class_weight[i];
  }
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
}

template <typename Dtype>
void MultiChannelBalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
	  const Dtype* class_weight = class_weight_.cpu_data();

    int dim = bottom[0]->height()*bottom[0]->width();

    for (int i = 0; i < num_label_; ++i) {
    	for (int j = 0; j < dim; ++j) {
	      int idx = i*dim+j;
        if (target[j] == (i+1)) {
			  	bottom_diff[idx] = sigmoid_output_data[idx] - 1;
          bottom_diff[idx] *= (1 - class_weight[i]);
        } else {
				  bottom_diff[idx] = sigmoid_output_data[idx] - 0;
         	bottom_diff[idx] *= class_weight[i]; 
				}
     	}
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MultiChannelBalancedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MultiChannelBalancedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiChannelBalancedSigmoidCrossEntropyLoss);

}  // namespace caffe
