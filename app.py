import streamlit as st
import time
import pickle
import sys
import numpy as np

from data_prep import input_feature

#st.set_page_config(layout="wide")

st.title('Stability and Magnetic Moment prediction for MXenes')
st.text('This is a web app to showcase our work related to property prediction for MXenes.')

with st.expander('About the Project'):
    st.subheader('MXenes')
    with st.container():
        col1, col2 = st.columns([7, 4])
        with col1:
            st.markdown('''
                MXenes are a class of two-dimensional inorganic compounds, that consist of atomically 
                thin layers of transition metal carbides, nitrides, or carbonitrides. They alo accept 
                a variety of hydrophilic terminations. They have the general formula of the form - 
                **M<sub>n+1</sub>X<sub>n</sub>T<sub>x</sub>**.
            ''', unsafe_allow_html=True)
        with col2:
            st.image('./mxene.jpg')
    st.markdown('''Here, M stands for transitions metals like Sc, Ti, V, etc. X can be either Carbon(C)
                or Nitrogen(N) and Tx are surface terminations like Cl, F, OH and O. The value of n varies
                from 1 to 3.''')
    st.markdown('''We have trained some ML models for MXene materials using the data from **C2DB database**. 
                The models are for stability classification and prediction of magnetic moments. After training
                on various classification and regression models, we got the best results on **Random Forest (AUC=0.89)**
                for stability, and on **XGBoost Regressor (R<sup>2</sup>=0.78)**.''', unsafe_allow_html=True)


if 'mxene_m' not in st.session_state:
    st.session_state.mxene_m = 'Sc'

if 'mxene_x' not in st.session_state:
    st.session_state.mxene_x = 'C'

if 'mxene_t' not in st.session_state:
    st.session_state.mxene_t = 'O'

if 'mxene_n' not in st.session_state:
    st.session_state.mxene_n = 1

if 'formula' not in st.session_state:
    st.session_state.formula = 'Sc2CO2'

if 'show_result' not in st.session_state:
    st.session_state.show_result = False

if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False

if 'mag_mom' not in st.session_state:
    st.session_state.mag_mom = ''

if 'stability' not in st.session_state:
    st.session_state.stability = ''



def fill_textbox1(value):
    st.session_state.mxene_m = value
    st.session_state.show_result = False
    update_formula()
def fill_textbox2(value):
    st.session_state.mxene_x = value
    st.session_state.show_result = False
    update_formula()
def fill_textbox3(value):
    st.session_state.mxene_t = value
    st.session_state.show_result = False
    update_formula()
def fill_textbox4(value):
    st.session_state.mxene_n = value
    st.session_state.show_result = False
    update_formula()

def update_formula():
    M = st.session_state.mxene_m
    X = st.session_state.mxene_x
    T = st.session_state.mxene_t
    n = st.session_state.mxene_n

    formula = M+str(n+1)+X+str(n)+T+'2'
    st.session_state.formula = formula

def show_result():
    st.session_state.show_result = True
    st.session_state.mag_mom = predict_mag_mom(st.session_state.formula)
    st.session_state.stability = predict_stability(st.session_state.formula)
    st.session_state.results_ready = False
    

def predict_stability(formula):
    with open('./stability/stability_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
    input = input_feature(formula)
    input = np.array(input).reshape(1,-1)
    input_scaled = scaler.transform(input)
    result = model.predict(input_scaled)
    return 'Stable' if result[0] else 'Not Stable'

def predict_mag_mom(formula):
    with open('./magnetic_moment/mag_mom_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
    input = input_feature(formula)
    input = np.array(input).reshape(1,-1)
    input_scaled = scaler.transform(input)
    result = model.predict(input_scaled)
    return result[0]

st.subheader('MXene: '+st.session_state.formula)

with st.container():
    col1, col2, col3, col4 = st.columns([4,2,2,1])
    with col1:
        textbox1 = st.empty()
        textbox1.text_input('M', st.session_state.mxene_m)
    with col2:
        textbox2 = st.empty()
        textbox2.text_input('X', st.session_state.mxene_x)
    with col3:
        textbox3 = st.empty()
        textbox3.text_input('T', st.session_state.mxene_t)
    with col4:
        textbox4 = st.empty()
        textbox4.text_input('n', st.session_state.mxene_n)



with st.container():
    col1, col2, col3, col4 = st.columns([4,2,2,1])
    with col1:
        with st.container():
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                st.button('Sc', on_click = lambda: fill_textbox1('Sc'), use_container_width=True)
            with col12:
                st.button('Ti', on_click = lambda: fill_textbox1('Ti'), use_container_width=True)
            with col13:
                st.button('V', on_click = lambda: fill_textbox1('V'), use_container_width=True)
            with col14:
                st.button('Cr', on_click = lambda: fill_textbox1('Cr'), use_container_width=True)
        with st.container():
            col21, col22, col23, col24 = st.columns(4)
            with col21:
                st.button('Y', on_click = lambda: fill_textbox1('Y'), use_container_width=True)
            with col22:
                st.button('Zr', on_click = lambda: fill_textbox1('Zr'), use_container_width=True)
            with col23:
                st.button('Nb', on_click = lambda: fill_textbox1('Nb'), use_container_width=True)
            with col24:
                st.button('Mo', on_click = lambda: fill_textbox1('Mo'), use_container_width=True)
        with st.container():
            col31, col32, col33, col34 = st.columns(4)
            with col31:
                st.button('Mn', on_click = lambda: fill_textbox1('Mn'), use_container_width=True)
            with col32:
                st.button('Hf', on_click = lambda: fill_textbox1('Hf'), use_container_width=True)
            with col33:
                st.button('Ta', on_click = lambda: fill_textbox1('Ta'), use_container_width=True)
            with col34:
                st.button('W', on_click = lambda: fill_textbox1('W'), use_container_width=True)

    with col2:
        with st.container():
            col11, col12, col13 = st.columns(3)
            with col11:
                pass
            with col12:
                st.button('C', on_click = lambda: fill_textbox2('C'), use_container_width=True)
            with col13:
                pass
        with st.container():
            col21, col22, col23 = st.columns(3)
            with col21:
                pass
            with col22:
                st.button('N', on_click = lambda: fill_textbox2('N'), use_container_width=True)
            with col23:
                pass
    with col3:
        with st.container():
            col11, col12 = st.columns(2)
            with col11:
                st.button('O', on_click = lambda: fill_textbox3('O'), use_container_width=True)
            with col12:
                st.button('F', on_click = lambda: fill_textbox3('F'), use_container_width=True)
        with st.container():
            col21, col22 = st.columns(2)
            with col21:
                st.button('Cl', on_click = lambda: fill_textbox3('Cl'), use_container_width=True)
            with col22:
                st.button('OH', on_click = lambda: fill_textbox3('OH'), use_container_width=True)
    
    with col4:
        with st.container():
            st.button('1', on_click = lambda: fill_textbox4(1), use_container_width=True)
            st.button('2', on_click = lambda: fill_textbox4(2), use_container_width=True)
            st.button('3', on_click = lambda: fill_textbox4(3), use_container_width=True)


st.write('')
st.button('Predict', on_click=show_result, type='primary')

st.divider()

if st.session_state.show_result:
    st.subheader('Results: ')
    if not st.session_state.results_ready:
        with st.spinner('Preparing results ...'):
            time.sleep(4)
        st.session_state.results_ready = True
    if st.session_state.results_ready:
        st.markdown(""" 
            <style> .result-div {
                padding: 20px;
                background-color: #e5e9ea;
                border: 1px solid #e5e9ea;
                border-radius: 5px} 
            </style> """, unsafe_allow_html=True)
        st.markdown("""
            <div class="result-div">
                <p> <b>Stability Classification</b>: {0} </p>
                <p> <b>Magnetic Moment</b>: {1} </p>
            </div>
            """.format(st.session_state.stability, st.session_state.mag_mom), unsafe_allow_html=True)
else:
    pass

