import streamlit as st

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.title('Burmese Alphabet Recognition System')
st.markdown('''Burmese Alphabet Recognition System is a system to check 
Burmese handwritten alphabets. It has two pages.''')
# = '<h1 style="color:#F8E6BC;">Burmese Alphabet Recognition System</h1>'
#st.markdown(title, unsafe_allow_html=True)
st.header('canvas page')
st.text('Navigate canvas page to write alphabet')

st.header('word canvas page')
st.text('Navigate canvas page to write alphabets in a squence')