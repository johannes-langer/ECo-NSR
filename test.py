import argparse

def some_func(argv=None):
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument('--text', type=str)
    args = parser.parse_args(argv) if argv else parser.parse_args()
    print(args.text)

if __name__ == '__main__':
    args = ['--text', 'This is pre-coded.']
    some_func(argv=args)
    some_func()
