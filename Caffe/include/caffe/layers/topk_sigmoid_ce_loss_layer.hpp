#ifndef CAFFE_TOPK_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_TOPK_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class TopkSigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit TopkSigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TopkBinLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
	static int comp(const void *a, const void *b);
  // threshold to check the neighbouring pixels
  bool has_ignore_label_;
  int ignore_label_;
  int outer_num_, inner_num_;

  Dtype keep_ratio_; /* keep_negative_pixels / all_negative_pixels */
  Dtype keep_thres_; /* ignore small loss values */
  int count_pos_, count_neg_; /* the number of positive and negative pixels */
  Dtype weight_pos_, weight_neg_; 
  Dtype thres_neg_; /* loss value of the num_keep_neg_^th largest negative pixel */
  int num_keep_neg_; /* the number of netagive pixels kept */
  Blob<Dtype> neg_loss_; /* keep loss value of negative pixels */

  int iter_; /* number of forward_cpu performed */
//  bool schedule_dynamic_; /* schedule keep_ratio_ dynamically */
  TopkSigmoidCELossParameter_ScheduleType schedule_type_;
};

}  // namespace caffe

#endif  // CAFFE_TOPK_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
