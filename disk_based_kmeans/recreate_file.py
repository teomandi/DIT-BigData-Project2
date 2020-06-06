from utils import recreate_file
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--tags',
                        type=str,
                        help="The path for the tags file",
                        action='store',
                        required=True)
    parser.add_argument('-m',
                        '--movies',
                        type=str,
                        help="The path for the movies file",
                        action='store',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help="The path to store the recreated file",
                        action='store',
                        required=True)
    args = parser.parse_args()
    arguments = vars(args)
    print(arguments)

    recreate_file(tags_path=arguments['tags'], movies_path=arguments['movies'], output_path=arguments['output'])



