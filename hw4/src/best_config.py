import basic_features
import alignment_features
import brown_clusters
import dwl_feature

FEATURES = [basic_features.n_oov, # 1
            basic_features.n_target, # 1
            basic_features.n_target_type, # 4
            basic_features.ef_ratio, # 1
            basic_features.log_ef_ratio, # 1
            alignment_features.dist_2_diag, # 1
            alignment_features.jump_dist, # 6
            alignment_features.fertilities, # 8
            brown_clusters.lm_score, # 1
            brown_clusters.tm_score, # 1
            dwl_feature.get_dwl_prob] # 1
