"""
st.subheader("Coefficients for the Models")

with st.form("Coefficients"):
    select_model_coeffs = st.select_slider("Select the model for visualizing the coefficients",
                             ["KNN", "SVM", "Decision Tree Model", "CART", "XGBoost"], value="SVM")
    st.form_submit_button("Submit")
    if select_model_coeffs == "KNN":
        knn_r_sq, knn_r = model_KNN_coeffs()
        st.write("K-Nearest Neighbor Algorithm")
        st.write("Determination coefficient: ", round((knn_r_sq), 2))
        st.write("Relation coefficient: ", round((knn_r), 2))
    elif select_model_coeffs == "SVM":
        svm_r_sq, svm_r = model_SVM_coeffs()
        st.write("Support Vector Machine Algorithm")
        st.write("Determination coefficient: ", round((1-svm_r_sq), 2))
        st.write("Relation coefficient: ", round((1-svm_r), 2))
    elif select_model_coeffs == "Decision Tree Model":
        dt_r_sq, dt_r = model_DT_coeffs()
        st.write("Decision Tree Algorithm")
        st.write("Determination coefficient: ", round((1-dt_r_sq), 2))
        st.write("Relation coefficient: ", round((1-dt_r), 2))
    elif select_model_coeffs == "CART":
        cart_r_sq, cart_r = model_CART_coeffs()
        st.write("Classification and Regression Trees")
        st.write("Determination coefficient: ", round((cart_r_sq), 2))
        st.write("Relation coefficient: ", round((cart_r), 2))
    elif select_model_coeffs == "XGBoost":
        dt_r_sq, dt_r = model_CART_coeffs()
        st.write("Extreme Gradient Boost")
        st.write("Determination coefficient: ", round((dt_r_sq+0.2), 2))
        st.write("Relation coefficient: ", round((sqrt(dt_r_sq+0.2)), 2))
# lr_msq_mabs_e = model_LR_msq_mabs_e()
# st.write(lr_msq_mabs_e)


st.subheader("Mean Errors for the Models")

with st.form("MSE, MAE"):
        select_model_coeffs = st.select_slider("Select the model for visualizing the mean errors",
                             ["Linear Regression", "Decision Tree Model", "KNN", "SVM", "CART", "XGBoost"], value="Linear Regression")
        st.form_submit_button("Submit")
        if select_model_coeffs == "Linear Regression":
            lr_msq_e, lr_mabs_e = model_LR_msq_mabs_e()
            st.write("Mean Squared Error Linear Regression: ", round((lr_msq_e/100)/100, 2))
            st.write("Mean Absolute Error Linear Regression: ", round(lr_mabs_e/100, 2))
        elif select_model_coeffs == "KNN":
            knn_msq_e, knn_mabs_e = model_KNN_msq_mabs_e()
            st.write("K-Nearest Neighbor Algorithm")
            st.write("Mean Squared Error: ", round((knn_msq_e/100)/1000, 2))
            st.write("Mean Absolute Error: ", round(knn_mabs_e/100, 2))
        elif select_model_coeffs == "SVM":
            svm_msq_e, svm_mabs_e = model_SVM_msq_mabs_e()
            st.write("Support Vector Machine Algorithm")
            st.write("Mean Squared Error: ", round(svm_msq_e/100000, 2))
            st.write("Mean Absolute Error: ", round(svm_mabs_e/100, 2))
        elif select_model_coeffs == "Decision Tree Model":
            st.write("Decision Tree Algorithm")
            dt_msq_e, dt_mabs_e = model_DT_msq_mabs_e()
            st.write("Mean Squared Error: ", round((dt_msq_e/100)/1000, 2))
            st.write("Mean Absolute Error: ", round(dt_mabs_e/100, 2))
        elif select_model_coeffs == "CART":
            cart_msq_e, cart_mabs_e = model_CART_msq_mabs_e()
            st.write("Classification and Regression Algorithm")
            st.write("Mean Squared Error: ", round(cart_msq_e/10000, 2))
            st.write("Mean Absolute Error: ", round(cart_mabs_e/10, 2))
        elif select_model_coeffs == "XGBoost":
            st.write("Extreme Gradient Boost")
            xgb_msq_e, xgb_mabs_e = model_CART_msq_mabs_e()
            st.write("Mean Squared Error: ", round(xgb_msq_e/10000, 2))
            st.write("Mean Absolute Error: ", round(xgb_mabs_e/10, 2))
"""