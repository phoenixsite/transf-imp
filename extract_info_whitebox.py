

import os

from collections import OrderedDict
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Extract information from the ACG and ReACG stats files.",
    )
    parser.add_argument('dirpath', type=str)
    args = parser.parse_args()

    models_dir = os.scandir(args.dirpath)
    
    eps_mapping = {"eps1": {}, "eps2": {}, "eps3": {}, "eps4": {}}
    asrs = OrderedDict(eps_mapping)
    nimages = OrderedDict(eps_mapping)

    # order_target = [
    #     "vgg16.tv_in1k",
    #     "vgg19.tv_in1k",
    #     "resnet50.tv2_in1k",
    #     "resnet152.tv2_in1k",
    #     "inception_v3.tv_in1k",
    #     "mobilenetv2_140.ra_in1k",
    # ]
    order_target = [
        "random-padding",
        "bit-reduction",
        "jpeg",
        "neural-representation-purifier",
        "inception_v3.tf_adv_in1k",
        "inception_resnet_v2.tf_ens_adv_in1k",
    ]
    order_attack = ["APGD", "ACG", "ReACG"]
    order_base = ["VGG16", "Res50", "Inc-v3"]

    for iter in asrs.keys():
        for base_model in order_base:
            asrs[iter].update({base_model: {}})
            nimages[iter].update({base_model: {}})
            for attack in order_attack:
                asrs[iter][base_model].update({attack: {}})
                nimages[iter][base_model].update({attack: {}})
                for target_model in order_target:
                    asrs[iter][base_model][attack].update({target_model: None})
                    nimages[iter][base_model][attack].update({target_model: None})

    for model_dir in models_dir:

        target_model = model_dir.name

        for f in os.scandir(model_dir):
            if ".txt" in f.name:
                file = f
                break

        with open(file) as f:
            while f:

                for iter in asrs.keys():
                    first = f.readline()

                    if first:
                        base_model = None

                        if "inception" in first:
                            base_model = "Inc-v3"
                        elif "resnet" in first:
                            base_model = "Res50"
                        elif "vgg16" in first:
                            base_model = "VGG16"

                        if not base_model:
                            raise RuntimeError("base model not found")

                        attack = None

                        if "ReACG" in first:
                            attack = "ReACG"
                        elif "APGD" in first:
                            attack = "APGD"
                        elif "ACG" in first:
                            attack = "ACG"

                        if not attack:
                            raise RuntimeError("attack not found")

                        second = f.readline()
                        nimage = int(second.split(":")[-1].strip())
                        nimages[iter][base_model][attack][target_model] = nimage
                        _ = f.readline()
                        _ = f.readline()

                        last = f.readline()

                        asr = last.split("=")[-1].strip()
                        asr = float(asr)
                        _ = f.readline()
                        _ = f.readline()
                        asrs[iter][base_model][attack][target_model] = asr
                
                if not first:
                    break

    for iter in asrs.keys():
        print(f"\nTABLE {iter}---------------------------------------------------------------------------------")
        for base_model in order_base:

            msg = "\multirow{3}{4em}{" + base_model + "}" + f" & {order_attack[0]} & "
            for target_model in order_target[:-1]:
                msg += f"{asrs[iter][base_model][order_attack[0]][target_model]:.2f} & "

            msg += f"{asrs[iter][base_model][order_attack[0]][order_target[-1]]:.2f} \\\ "
            print(msg)

            for attack in order_attack[1:]:
                msg = f"& {attack} & "
                for target_model in order_target[:-1]:
                    msg += f"{asrs[iter][base_model][attack][target_model]:.2f} & "

                msg += f"{asrs[iter][base_model][attack][order_target[-1]]:.2f} \\\[1em] "

                print(msg)

    print("\n\n--------------------------------------------\n\n")
    for iter in nimages.keys():
        print(f"TABLE {iter}---------------------------------------------------------------------------------\n")
        for base_model in order_base:

            msg = "\multirow{3}{4em}{\makecell{" + base_model + " \\\ \scriptsize{(\%)}}}" + " & \makecell{" + order_attack[0] + " \\\ \scriptsize{(\%)}} & "
            for target_model in order_target[:-1]:
                msg += "\makecell{" + str(nimages[iter][base_model][order_attack[0]][target_model]) + " \\\ \scriptsize{(\%)}} & "

            msg += "\makecell{" + str(nimages[iter][base_model][order_attack[0]][order_target[-1]]) + " \\\ \scriptsize{(\%)}} \\\ "
            print(msg)

            for attack in order_attack[1:]:
                msg = "& \makecell{" + attack + " \\\ \scriptsize{(\%)}} & "
                for target_model in order_target[:-1]:
                    msg += "\makecell{" + str(nimages[iter][base_model][attack][target_model]) + " \\\ \scriptsize{(\%)}} & "

                msg += "\makecell{" + str(nimages[iter][base_model][attack][order_target[-1]]) + " \\\ \scriptsize{(\%)}} \\\[2em] "

                print(msg)
                    
