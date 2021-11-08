import pickle
import numpy as np
import streamlit as st
from PIL import Image

model=pickle.load(open('model.pkl','rb'))


def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():

    st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.pexels.com/photos/51951/forest-fire-fire-smoke-conservation-51951.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")
    }
   .sidebar .sidebar-content {
        background: url("https://images.pexels.com/photos/51951/forest-fire-fire-smoke-conservation-51951.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.title("Forest Fire")
    html_temp = """
    <div style="background color:#66a3ff ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    oxygen = st.text_input("Oxygen","Type Here")
    humidity = st.text_input("Humidity","Type Here")
    temperature = st.text_input("Temperature","Type Here")



    safe_html="""  
      <div style="background-color:#e600ac;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#0080ff;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(oxygen,humidity,temperature)
        st.success('## The probability of fire taking place is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
