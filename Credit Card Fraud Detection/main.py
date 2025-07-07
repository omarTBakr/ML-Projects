from src.utils import load_model, visualize_all, load_split_test_splitted

if __name__ == "__main__":
    model = load_model("HardVoting")
    x_test, y_test =load_split_test_splitted()


    visualize_all(model, x_test, y_test)
