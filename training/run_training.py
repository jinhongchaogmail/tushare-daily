import os
import sys
import runpy

def main():
    script_path = os.path.join(os.path.dirname(__file__), '模型训练.py')
    # Preserve CLI args for the target script
    sys.argv = [script_path] + sys.argv[1:]
    runpy.run_path(script_path, run_name='__main__')

if __name__ == '__main__':
    main()
