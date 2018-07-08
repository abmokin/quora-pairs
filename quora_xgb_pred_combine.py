from quora_utils import *
import time

def main():
    # Start timer
    start_time = time.time()

    # Combine prediction files
    with open(get_current_path() + '/predictions/pred1.csv', 'w') as outfile:
        outfile.write('test_id,is_duplicate\n')
        for file_num in range(1,25):
            with open(get_current_path() + '/pred/pred_' + str(file_num) + '.csv') as infile:
                for line in infile:
                    outfile.write(line)

    # Display the time of program execution
    print('--- {:.2f} minutes ---'.format((time.time() - start_time)/60))

if __name__ == "__main__":
    main()