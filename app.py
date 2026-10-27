
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Energy Load Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    data = np.load("outputs/data_split.npz", allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    feature_names = data["feature_names"].tolist()

    model = DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train, feature_names


model, X_train, feature_names = load_model()

st.title("🏠 Building Energy Load Predictor")
st.write(
    "Estimate a building's heating load using a trained "
    "machine learning model."
)

st.info(
    "Enter the building's eight design features below, "
    "then click Predict Heating Load."
)

st.sidebar.header("About this model")
st.sidebar.write("Model: Decision Tree Regressor")
st.sidebar.write("Maximum tree depth: 5")
st.sidebar.write("Target: Heating Load (Y1)")
st.sidebar.write("Validation MAE: 0.6904")
st.sidebar.write("Held-out test MAE: 0.7453")

st.subheader("Building Design Features")

descriptions = {
    "relative_compactness": "Relative compactness",
    "surface_area": "Surface area",
    "wall_area": "Wall area",
    "roof_area": "Roof area",
    "overall_height": "Overall height",
    "orientation": "Orientation (2–5)",
    "glazing_area": "Glazing area",
    "glazing_area_distribution": "Glazing area distribution (0–5)"
}

integer_features = {
    "orientation",
    "glazing_area_distribution"
}

input_values = {}

left, right = st.columns(2)

for index, feature in enumerate(feature_names):
    column = left if index % 2 == 0 else right

    minimum = float(np.min(X_train[:, index]))
    maximum = float(np.max(X_train[:, index]))
    default = float(np.median(X_train[:, index]))

    label = descriptions.get(
        str(feature),
        str(feature).replace("_", " ").title()
    )

    with column:
        if feature in integer_features:
            input_values[feature] = st.selectbox(
                label,
                options=list(range(
                    int(np.ceil(minimum)),
                    int(np.floor(maximum)) + 1
                )),
                index=max(
                    0,
                    min(
                        len(range(
                            int(np.ceil(minimum)),
                            int(np.floor(maximum)) + 1
                        )) - 1,
                        int(round(default)) -
                        int(np.ceil(minimum))
                    )
                )
            )
        else:
            input_values[feature] = st.slider(
                label,
                min_value=minimum,
                max_value=maximum,
                value=float(np.clip(default, minimum, maximum)),
                step=(maximum - minimum) / 100
                if maximum > minimum else 0.01
            )

st.divider()

if st.button("Predict Heating Load", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[input_values[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = float(model.predict(input_df)[0])

    st.success("Prediction completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Heating Load",
            f"{prediction:.2f}"
        )

    with col2:
        st.metric(
            "Model",
            "Decision Tree"
        )

    st.caption(
        "The prediction is expressed in the dataset's heating-load "
        "units. It is an estimate, not a guarantee of real-world "
        "building performance."
    )

    st.subheader("Your Building Inputs")
    st.dataframe(
        input_df.T.rename(columns={0: "Selected value"}),
        use_container_width=True
    )

st.divider()
st.caption(
    "Educational machine learning project | "
    "Dataset: UCI Energy Efficiency"
)
