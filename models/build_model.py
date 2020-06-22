from models import resnet, pyconvresnet, pyconvhgresnet


def build_model(args):

    if args.arch == 'pyconvhgresnet':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = pyconvhgresnet.pyconvhgresnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = pyconvhgresnet.pyconvhgresnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = pyconvhgresnet.pyconvhgresnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'pyconvresnet':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = pyconvresnet.pyconvresnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = pyconvresnet.pyconvresnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = pyconvresnet.pyconvresnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'resnet':
        assert args.model_depth in [18, 34, 50, 101, 152, 200]

        if args.model_depth == 18:
            model = resnet.resnet18(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = resnet.resnet34(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = resnet.resnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resnet.resnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resnet.resnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resnet.resnet200(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    return model
