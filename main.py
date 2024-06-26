from harmonybatch import App, Apps, FunCfg, get_config, Harmony

if __name__ == "__main__":
    config = get_config()
    # f = FunCfg(config)
    apps_list = [App("app0", 0.2, 5), App("app1", 0.3, 5), App("app2", 0.4, 5)]
    # cfg = f.get_config(Apps(apps_list))
    # print(cfg)
    hb = Harmony(config)
    cfgs = hb.get_config(Apps(apps_list))
    print("Provisioning plan:")
    for i in range(0, len(cfgs)):
        print("The configurations of the group", i ,"is:")
        print(cfgs[i])