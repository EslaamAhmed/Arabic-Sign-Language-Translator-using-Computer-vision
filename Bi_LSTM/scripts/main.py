from scripts import train, predict, collect_data


def main():
    print("Select mode: 'train' or 'predict'")
    mode = input("Enter mode: ").strip().lower()
    
    if mode == 'train':
        train.run_training()
    elif mode == 'predict':
        predict.predict_actions()
    elif mode == 'collect data':
        collect_data.collect_data()
    else: 
        print("Invalid mode selected. Please choose 'train', 'predict' or 'collect data'.")

if __name__ == "__main__":
    main()

