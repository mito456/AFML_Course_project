import unittest
from adapt_methods import few_label_recalibrate, no_label_rank_adjust

SCORES = [{"video_name":"a","predicted_probability":.2},{"video_name":"b","predicted_probability":.8},{"video_name":"c","predicted_probability":.5},{"video_name":"d","predicted_probability":.4}]
K1 = {"a":"sp1", "b":"sp2"}

class AdaptMethodsTests(unittest.TestCase):
    def call(self, labels, **kwargs):
        return few_label_recalibrate(SCORES, labels, k=1, seed=0, evaluation_ids=["c","d"], expected_adaptation_ids=K1, source_prior=.4, **kwargs)
    def test_k0_rank_not_calibration(self):
        self.assertEqual(no_label_rank_adjust(SCORES)[0]["score_kind"], "unsupervised_target_midrank")
    def test_explicit_source_prior_and_distribution_distinction(self):
        labels=[{"video_name":"a","usefulness":1},{"video_name":"b","usefulness":1}]
        low=self.call(labels); high=few_label_recalibrate(SCORES,labels,k=1,seed=0,evaluation_ids=["c","d"],expected_adaptation_ids=K1,source_prior=.8)
        self.assertNotEqual(low,high)
        with self.assertRaises(TypeError): few_label_recalibrate(SCORES,labels,k=1,seed=0,evaluation_ids=["c"],expected_adaptation_ids=K1)
    def test_all_seeds_and_reproducibility(self):
        labels=[{"video_name":"a","usefulness":1},{"video_name":"b","usefulness":0}]
        for seed in range(5): self.assertEqual(few_label_recalibrate(SCORES,labels,k=1,seed=seed,evaluation_ids=["c","d"],expected_adaptation_ids=K1,source_prior=.4),few_label_recalibrate(SCORES,labels,k=1,seed=seed,evaluation_ids=["c","d"],expected_adaptation_ids=K1,source_prior=.4))
        k3_scores = [{"video_name":str(i),"predicted_probability":.2 + i / 20} for i in range(7)]
        k3_ids = {"0":"sp1", "1":"sp1", "2":"sp1", "3":"sp2", "4":"sp2", "5":"sp2"}
        k3_labels = [{"video_name":str(i),"usefulness":i % 2} for i in range(6)]
        self.assertEqual(1, len(few_label_recalibrate(k3_scores,k3_labels,k=3,seed=4,evaluation_ids=["6"],expected_adaptation_ids=k3_ids,source_prior=.4)))
    def test_protocol_and_overlap_guards(self):
        labels=[{"video_name":"a","usefulness":1},{"video_name":"b","usefulness":0}]
        with self.assertRaises(ValueError): few_label_recalibrate(SCORES,labels,k=1,seed=0,evaluation_ids=["c"],expected_adaptation_ids=None,source_prior=.4)
        with self.assertRaises(ValueError): few_label_recalibrate(SCORES,labels,k=1,seed=0,evaluation_ids=["c"],expected_adaptation_ids={"a":"sp1","b":"sp1"},source_prior=.4)
        with self.assertRaises(ValueError): self.call(labels+[labels[0]])
        with self.assertRaises(ValueError): few_label_recalibrate(SCORES,labels,k=1,seed=0,evaluation_ids=["a"],expected_adaptation_ids=K1,source_prior=.4)
    def test_invalid_probability_and_prohibited_fields(self):
        with self.assertRaises(ValueError): no_label_rank_adjust(SCORES+[SCORES[0]])
        with self.assertRaises(ValueError): no_label_rank_adjust([{"video_name":"x","predicted_probability":2}])
        with self.assertRaises(ValueError): no_label_rank_adjust([{"video_name":"x","predicted_probability":.2,"audio_emb":[]}])
        with self.assertRaises(ValueError): self.call([{"video_name":"a","usefulness":1,"species":"x"}])

if __name__ == "__main__": unittest.main()
