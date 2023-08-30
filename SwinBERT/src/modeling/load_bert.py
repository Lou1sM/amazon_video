from src.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning

def get_bert_model(do_lower_case):
    # Load pretrained bert and tokenizer based on training configs
    config = BertConfig.from_pretrained('SwinBERT/models/captioning/bert-base-uncased/', num_labels=2, finetuning_task='image_captioning')

    tokenizer = BertTokenizer.from_pretrained('SwinBERT/models/captioning/bert-base-uncased/', do_lower_case=do_lower_case)
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = 0.1
    config.loss_type = 'classification'
    config.tie_weights = False # tie decoding and encoding weights
    config.freeze_embedding = False # freeze word embs
    config.label_smoothing = 0.
    config.drop_worst_ratio = 0.
    config.drop_worst_after = 0.
    config.img_feature_dim = 512
    model = BertForImageCaptioning(config=config) # init from scratch
    # update model structure if specified in arguments
    #update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    #model_structure_changed = [False] * len(update_params)
    ## model_structure_changed[0] = True  # cclin hack
    #for idx, param in enumerate(update_params):
    #    arg_param = getattr(args, param)
    #    # bert-base-uncased do not have img_feature_dim
    #    config_param = getattr(config, param) if hasattr(config, param) else -1
    #    if arg_param > 0 and arg_param != config_param:
    #        setattr(config, param, arg_param)
    #        model_structure_changed[idx] = True
    #if any(model_structure_changed):
    #    assert config.hidden_size % config.num_attention_heads == 0
    #    if args.load_partial_weights:
    #        # can load partial weights when changing layer only.
    #        assert not any(model_structure_changed[2:]), "Cannot load partial weights " \
    #            "when any of ({}) is changed.".format(', '.join(update_params[2:]))
    #        model = model_class.from_pretrained(args.model_name_or_path,
    #            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    #    else:
    #        model = model_class(config=config) # init from scratch
    #else:
    #    model = model_class.from_pretrained(args.model_name_or_path,
    #        from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    #total_params = sum(p.numel() for p in model.parameters())
    return model, config, tokenizer
