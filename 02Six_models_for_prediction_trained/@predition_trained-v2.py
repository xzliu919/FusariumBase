from PNNGS_prediction_trained import fusarium_PNNGS
from ResGS_prediction_trained import fusarium_ResGS
from machine_learning_prediction_trained import fusarium_machine_learning
import argparse


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='病原对寄主植物致病性预测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 添加参数定义
    parser.add_argument(
        '-p', '--pheno',
        type=str,
        choices=["wheat_stem", "wheat_head", "Maize_stem", "Soybean_stem"],
        default="wheat_head",
        help='选择表型类型，可选：%(choices)s'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default="input_OG_group.txt",
        help='输入文件路径（Orthogroups TSV文件）'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=["PNNGS", "ResGS", "GradientBoostingRegressor",
                 "RandomForestRegressor", "Ridge", "SVR"],
        default="RandomForestRegressor",
        help='选择预测模型，可选：%(choices)s'
    )

    # 解析参数
    args = parser.parse_args()

    # 打印参数信息
    print(f"当前配置参数：")
    print(f"  - 表型类型: {args.pheno}")
    print(f"  - 输入文件: {args.input}")
    print(f"  - 预测模型: {args.model}\n")

    # 执行预测流程
    try:
        if args.model == "PNNGS":
            output = fusarium_PNNGS(args.input, args.pheno)
        elif args.model == "ResGS":
            output = fusarium_PNNGS(args.input, args.pheno)
            fusarium_ResGS(args.input, args.pheno)
        else:
            output = fusarium_machine_learning(args.input, args.pheno, args.model)

        print("预测结果：", output)
    except Exception as e:
        print(f"运行时错误：{str(e)}")
        exit(1)


if __name__ == '__main__':
    main()