#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_randreplace_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidRandReplaceCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const SigmoidRandReplaceCELossParameter  randrep_loss_param = this->layer_param_.randrep_loss_param();
  schedule_type_ = randrep_loss_param.schedule_type();
}

template <typename Dtype>
void SigmoidRandReplaceCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_RANDREPLACE_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->num(), 1) <<
      "SIGMOID_RANDREPLACE_CROSS_ENTROPY_LOSS layer only supports batchsize 1";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  weight_matrix_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SigmoidRandReplaceCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  Dtype* prob_data = sigmoid_output_->mutable_cpu_data();
  Dtype* weight_mat = weight_matrix_.mutable_cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype count_pos = 0;
  Dtype count_neg = 0;
  Dtype loss = 0;

  int dim = bottom[0]->count() / bottom[0]->num();

  /* step 1: count positive and negative samples */
  for (int j = 0; j < dim; j++) {
	if (target[j] == 1) {
	  count_pos++;
	} else if (target[j] == 0) {
	  count_neg++;
	} else {/* */}
  }

  /* step 2: random drop background pixel */
  for (int j = 0; j < dim; j++) {
	if (target[j] == 0) {
	  Dtype rand_temp, drop_prob;
      caffe_rng_uniform(int(1), Dtype(0), Dtype(1), &rand_temp);
      switch (schedule_type_) {
        case SigmoidRandReplaceCELossParameter_ScheduleType_LINEAR:
          drop_prob = (1.0 - prob_data[j]);
          break;
        case SigmoidRandReplaceCELossParameter_ScheduleType_SQU:
          drop_prob = (1.0-prob_data[j])*(1.0-prob_data[j]);
          break;
        case SigmoidRandReplaceCELossParameter_ScheduleType_LOG:
          drop_prob = (1.0 + log10(1.0 - prob_data[j]));
          break;
        default:
          drop_prob = (1.0 - prob_data[j]);
      }
      if (rand_temp < drop_prob) { 
		prob_data[j] = 0;
		count_neg--;
	  }
	}
  }

  /* step 3: assignment weight_matrix  */
  for (int j = 0; j < dim; j++) {
	if (target[j] == 1) {
	  weight_mat[j] = (count_neg) / (count_neg + count_pos);
	} else if (target[j] == 0) {
	  if (prob_data[j] > 0) {
        weight_mat[j] = (count_pos) / (count_neg + count_pos); 
	  } else {
		weight_mat[j] = 0;
	  }
	} else { /* */}
  }

  /* step 4: calculate negative log-likelihood */
  for (int j = 0; j < dim; j++) {
	Dtype pt = (target[j] == 1) ? prob_data[j] : (1.0 - prob_data[j]);
    loss += -weight_mat[j] * log(std::max(pt, Dtype(FLT_MIN)));
  }

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SigmoidRandReplaceCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    const Dtype* weight_mat = weight_matrix_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    int dim = bottom[0]->count() / bottom[0]->num();

   	for (int j = 0; j < dim; j ++) {
	  bottom_diff[j] *= weight_mat[j];
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidRandReplaceCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidRandReplaceCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidRandReplaceCrossEntropyLoss);

}  // namespace caffe
