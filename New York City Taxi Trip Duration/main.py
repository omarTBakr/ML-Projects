from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import linear_model
from utilities import load_train, load_val, transform_split,load_test
from modeling import train, evaluate, save_model


if __name__ == "__main__":

    val_data = load_val()
    train_data = load_train()

    model = linear_model.Ridge(alpha=1, fit_intercept=True)
    pipleline = Pipeline([ ("normalizer", MinMaxScaler()),
                           ("polynomial_features", PolynomialFeatures(degree=2)) ])

    x_train_featured, y_train_transformed = transform_split(train_data)
    x_train_transformed = pipleline.fit_transform(x_train_featured)
    # training the model
    train(model, x_train_transformed, y_train_transformed)
    # evaluate the model
    print("================Evaluating over Train Data  =============")
    evaluate(model, x_train_transformed, y_train_transformed)

    x_val_featured, y_val_transformed = transform_split(val_data)
    x_val_transformed = pipleline.transform(x_val_featured)

    print("================Evaluating over Val Data  =============")
    val_dict = evaluate(model, x_val_transformed, y_val_transformed)

    if val_dict["r2_score"] >= 0.69:
        save_model(model)


    test_data = load_test()
    print("================Evaluating over Test Data  =============")
    x_test_featured, y_test_transformed = transform_split(test_data)
    x_test_transformed = pipleline.transform(x_test_featured)

    print("================Evaluating over Val Data  =============")
    test_dict = evaluate(model, x_test_transformed, y_test_transformed)

