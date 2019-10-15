import sys
from preprocess.financial import pre_process_loan_data

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename and data set to pre-process")
    else:
        data_set = sys.argv[1]
        if data_set == 'cardio':
            pass
            # pre_process_cardio_data('')
        elif data_set == 'loan':
            pre_process_loan_data('')
