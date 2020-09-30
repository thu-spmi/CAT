import torch
import argparse
import collections

def main():
    parser = argparse.ArgumentParser(description="convert argument")
    parser.add_argument("--input_model", type=str, help='input model file name')
    parser.add_argument("--output_model", type=str, help='output model file name')
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()

    model = torch.load(args.input_model)['model']
    torch.save(model, '{}'.format("model.pt"))
    #print(model)
    reg_model = collections.OrderedDict()
    for i in range(args.layers):
        print("layer {}".format(i))
        reg_model['net.lstm{}.weight_ih_l0'.format(i)] = model['net.lstm1.weight_ih_l{}'.format(i)]
        reg_model['net.lstm{}.weight_hh_l0'.format(i)] = model['net.lstm1.weight_hh_l{}'.format(i)]
        reg_model['net.lstm{}.bias_ih_l0'.format(i)] = model['net.lstm1.bias_ih_l{}'.format(i)]
        reg_model['net.lstm{}.bias_hh_l0'.format(i)] = model['net.lstm1.bias_hh_l{}'.format(i)]
        reg_model['net.lstm{}.weight_ih_l0_reverse'.format(i)] = model['net.lstm1.weight_ih_l{}_reverse'.format(i)]
        reg_model['net.lstm{}.weight_hh_l0_reverse'.format(i)] = model['net.lstm1.weight_hh_l{}_reverse'.format(i)]
        reg_model['net.lstm{}.bias_ih_l0_reverse'.format(i)] = model['net.lstm1.bias_ih_l{}_reverse'.format(i)]
        reg_model['net.lstm{}.bias_hh_l0_reverse'.format(i)] = model['net.lstm1.bias_hh_l{}_reverse'.format(i)]
    reg_model['linear.weight'] = model['linear.weight']
    reg_model['linear.bias'] = model['linear.bias']

    #print(reg_model)
    torch.save(reg_model, '{}'.format(args.output_model))

if __name__ == "__main__":
    main()
