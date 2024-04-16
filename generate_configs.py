from argparse import ArgumentParser

from localconfig import config

parser = ArgumentParser()
parser.add_argument("-hc", "--hidden_channels", nargs="+", help="Number of hidden channels")
parser.add_argument("-hn", "--hidden_neurons", nargs="+", help="Number of hidden neurons")
parser.add_argument("-n", "--name_addon", type=str, help="The name to add on config", default="")
args = parser.parse_args()

config.read("config")

for hc in args.hidden_channels:
    for hn in args.hidden_neurons:
        config.network.hidden_channels = hc
        config.network.hidden_neurons = hn
        config.save("config" + args.name_addon + str(hc) + "_" + str(hn))
