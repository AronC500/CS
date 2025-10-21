import os
import json
import argparse



def get_org(pathlist, jsonpath):
    if not os.path.exists(jsonpath):
        os.mkdir(jsonpath)
    for path in pathlist:
        for file in os.scandir(path):
            f = open(file.path, 'r')
            blocks = {}
            block = {"insn_list": [], "out_edge_list": []}
            id = -1
            for line in f.readlines():
                if line.strip() == '':
                    break

                elements = line.strip().split(', ')
                if len(elements) < 4:
                    print('error: ' + file.name)
                    break

                if id == -1:
                    id = int(elements[0])

                if int(elements[0]) != id:
                    blocks[id] = block
                    block = {"insn_list": [], "out_edge_list": []}
                    id = int(elements[0])

                block['insn_list'].append([elements[1], elements[2], ', '.join(elements[3:])])
                if elements[2] == 'REF':
                    block['out_edge_list'].append(int(elements[3]))
            blocks[id] = block

            josonfp = os.path.join(jsonpath, file.name)
            jsonf = open(josonfp, "w")
            json.dump(blocks, jsonf, indent=4)
            jsonf.close()


def main():
    parser = argparse.ArgumentParser(description='Extract ORG for PDF')
    parser.add_argument('-i', required=True, dest='ir_dir', action='store', help='Input IR files dir' )
    parser.add_argument('-o', required=True, dest='out_dir', action='store', help='Output ORG json files dir' )

    args = parser.parse_args()
    pathlist = [args.ir_dir]
    get_org(pathlist, args.out_dir)


if __name__ == '__main__':
    main()
