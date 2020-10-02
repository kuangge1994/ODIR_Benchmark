#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/topk_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void TopkSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }

  const TopkSigmoidCELossParameter  topk_sigmoid_ce_loss_param = this->layer_param_.topk_sigmoid_ce_loss_param();
  keep_thres_ = topk_sigmoid_ce_loss_param.thres();
  /* initialize keep_ratio_ */
  schedule_type_ = topk_sigmoid_ce_loss_param.schedule_type();
  if (schedule_type_ == TopkSigmoidCELossParameter_ScheduleType_FIXED) {
    keep_ratio_ = topk_sigmoid_ce_loss_param.ratio();
  } 
  iter_ = 0; /* 0 */
}

template <typename Dtype>
void TopkSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "TOPK_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->channels(), 1) <<
      "TOPK_SIGMOID_CROSS_ENTROPY_LOSS layer does not support batchsize > 1.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  neg_loss_.Reshape(1, 1, 1, bottom[0]->count());
}

template <typename Dtype>
int TopkSigmoidCrossEntropyLossLayer<Dtype>::comp(const void *a, const void *b) {
	return *(Dtype *)a < *(Dtype *)b ? 1 : -1;
}

template <typename Dtype>
void TopkSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype *neg_loss_arr = neg_loss_.mutable_cpu_data();
  Dtype *neg_loss_sort = neg_loss_.mutable_cpu_diff();

  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  //Dtype loss_neg_tail = 0;

  /* count loss pixel-by-pixel */
  int dim = bottom[0]->count() / bottom[0]->num();
  count_pos_ = 0;
  count_neg_ = 0;
  for (int j = 0; j < dim; j++) {
	if (target[j] == 1) {
	  count_pos_++;
	  loss_pos -= input_data[j] * (1 - (input_data[j] >= 0)) - log(1 + exp(input_data[j] - 2 * input_data[j] * (input_data[j] >= 0)));
	} else if (target[j] == 0) { /*  */
		neg_loss_arr[count_neg_++] = -(input_data[j] * (0 - (input_data[j] >= 0)) - log(1 + exp(input_data[j] - 2 * input_data[j] * (input_data[j] >= 0))));
	} else { /* ignore */ }
  } 

  /* copy neg_loss_arr to neg_loss_sort */
  caffe_copy(count_neg_, (const Dtype *)neg_loss_arr, neg_loss_sort);

  /* decreasing order: sort neg_loss_sort using quick sort */
  qsort(neg_loss_sort, count_neg_, sizeof(Dtype), comp);

  /* compute keep_ratio_ */
  if(schedule_type_ == TopkSigmoidCELossParameter_ScheduleType_STEP) {
	keep_ratio_ = 0.14 - 0.014 * floor(iter_*1.0/10000); /* step decay */
  } else if(schedule_type_==TopkSigmoidCELossParameter_ScheduleType_POLY){
	keep_ratio_ = 0.14 * ( 1 - iter_*1.0 / 100000); /* poly decay */
  } else {
    /* */
  }

  /* select the num_keep_neg^th largest loss */
  num_keep_neg_ = count_neg_;
  for (int j = 0; j < count_neg_; j++) {
    if (neg_loss_sort[j] < keep_thres_) { /* skip small loss values */
		num_keep_neg_ = j;
		break;
	} 
  }
  num_keep_neg_ = (int)(num_keep_neg_ * keep_ratio_) + 1; /* +1 to avoid underflow, that is to say, num_keep_neg_ equals 0 */
  thres_neg_ = neg_loss_sort[num_keep_neg_ - 1];
  for (int j = 0; j < num_keep_neg_; j++) {
    loss_neg += neg_loss_sort[j];
  }

  /* count loss_neg_tail 
  for (int j = num_keep_neg_; j < count_neg_; j++) {
	loss_neg_tail += 0.5 * neg_loss_sort[j] * neg_loss_sort[j];
  }*/

  weight_pos_ = 1.0 * num_keep_neg_ / (count_pos_ + num_keep_neg_);
  weight_neg_ = 1.0 * count_pos_ / (count_pos_ + num_keep_neg_);
  
  loss_pos *= weight_pos_;
  loss_neg *= weight_neg_;

  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
  //top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg + 0.01*loss_neg_tail);

  iter_++; /* update iter_ */
}

template <typename Dtype>
void TopkSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
	const Dtype *neg_loss_arr = neg_loss_.cpu_data();
	
	int k = -1;
	for (int j = 0; j < dim; ++j) {
		if (target[j] == 1) {
			bottom_diff[j] *= weight_pos_;
		} else if (target[j] == 0) {
			k++;
			if (neg_loss_arr[k] >= thres_neg_) {
			  bottom_diff[j] *= weight_neg_;
			} else {/* easy negative samples (tail) */
			  bottom_diff[j] = 0;
			 // bottom_diff[j] = sigmoid_output_data[j]*0.01;
			}
		} else {
			/* ignore */
		}
	}
    
    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TopkSigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(TopkSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(TopkSigmoidCrossEntropyLoss);

}  // namespace caffe
