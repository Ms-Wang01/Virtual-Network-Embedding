from argparse import ArgumentParser

def set_params():
    parser = ArgumentParser()
    parser.add_argument("--sn_fat_tree", action="store_true", default=False)
    parser.add_argument("--pods", default=4, type=int)

    parser.add_argument("--num_sn", default=100, type=int)
    
    parser.add_argument("--sn_barabasi_albert", action="store_true", default=False)
    parser.add_argument("--num_sn_m", default=4, type=int)

    parser.add_argument("--sn_erdos_renyi", action="store_true", default=False)
    parser.add_argument("--sl_prob", default=0.25, type=int)

    parser.add_argument("--sn_feat_low", default=50, type=int)
    parser.add_argument("--sn_feat_high", default=100, type=int)
    parser.add_argument("--sl_feat_low", default=50, type=int)
    parser.add_argument("--sl_feat_high", default=100, type=int)

    parser.add_argument("--num_vg_init", default=20, type=int)
    parser.add_argument("--num_vg_stream", default=40, type=int)

    parser.add_argument("--num_vg_rt_low", default=2000, type=int)
    parser.add_argument("--num_vg_rt_high", default=3000, type=int)
    
    parser.add_argument("--vg_interval", default=10, type=int)

    parser.add_argument("--num_vn_low", default=2, type=int)
    parser.add_argument("--num_vn_high", default=12, type=int)
    
    parser.add_argument("--vn_barabasi_albert", action="store_true", default=False)
    parser.add_argument("--num_vn_m", default=4, type=int)

    parser.add_argument("--vn_erdos_renyi", action="store_true", default=False)
    parser.add_argument("--vl_prob", default=0.5, type=int)

    parser.add_argument("--vn_feat_low", default=15, type=int)
    parser.add_argument("--vn_feat_high", default=30, type=int)
    parser.add_argument("--vl_feat_low", default=15, type=int)
    parser.add_argument("--vl_feat_high", default=30, type=int)


    parser.add_argument("--node_num_res", default=1, type=int)
    parser.add_argument("--link_num_res", default=1, type=int)
    
    parser.add_argument("--no_same_place", action="store_true", default=False)
    
    parser.add_argument("--seed", default=1398, type=int)

    args = parser.parse_args()
    return args