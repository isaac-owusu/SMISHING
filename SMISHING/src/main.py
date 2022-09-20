
import streamlit as st


# import subprocess, os
from helpers import make_predictions

from streamlit_autorefresh import st_autorefresh


def get_pred(msg):
    pred, pred_proba = make_predictions(msg)
    return pred, pred_proba

st.cache() 
def run():
    
    with open('../logs/message.txt', 'r') as f:
        msg = f.read()
        f.close()

    print(msg)

    return msg


# local styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


if __name__ == "__main__":


    # Run the autorefresh about every 2000 milliseconds (2 seconds) and stop
    # after it's been refreshed 100 times.

    count = st_autorefresh(interval=3000, limit=100, key="fizzbuzzcounter")

    st.title('Smshing')
    msg = run()

    sender = msg.split('>')[0]
    message = msg.split('>')[-1]

    st.write(f'[ sender ] > {sender} \n\n[ Message ] > {message}')

    # st.write(msg)
    pred, pred_proba = get_pred(message)
    st.write(f'[ prediction ] > {pred}') # see *
    st.write(f'[ Confidence of ] > {pred_proba}') # see *

    # stop run
    st.stop()

    


