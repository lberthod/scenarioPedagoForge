
# based on this github : https://github.com/nicknochnack/Langchain-Crash-Course
# Bring in deps
import os 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
    
os.environ['OPENAI_API_KEY'] = st.secrets["auth"]

# App framework
st.set_page_config(
   page_title="scenarioPedagoForge",
   page_icon="🧊",
)


# Titre principal
st.title('📚 scenarioPedagoForge    ')

# Description
st.write('📚 GÉNÉRATEUR DE SCÉNARIO PÉDAGOGIQUE')
st.write("Cet outil est conçu pour aider les enseignants à élaborer un scénario pédagogique pour leurs cours à partir d'une thématique de matière.")
st.info("Veuillez fournir les informations ci-dessous afin de générer le contenu le plus adapté à vos besoins. Toutes les informations sont facultatives mais contribuent à améliorer la précision de la sortie générée.")

# Sous-titre
st.title('Décrivez votre cours')

# Zone de saisie pour la thématique
st.info("Dans cette zone de saisie, vous pouvez fournir des informations sur votre cours et votre matière pour aider à élaborer un scénario pédagogique adapté.")
prompt_title = st.text_input('Thématique de cours') 

# Slider pour choisir la durée du cours
st.write("Nombre de sections)")
numberQ = st.slider("Sélectionnez le nombre de sous-sections", 1, 12, 1)

# Suite du code pour traiter les informations et générer le scénario pédagogique...


# Prompt templates
title_template = PromptTemplate(
    input_variables = ['prompt_title' , 'numberQ'], 
    template = "En tant qu'expert en enseignement et en pédagogie, vous êtes chargé(e) de concevoir un cours sur la thématique suivante : {prompt_title} . Veuillez transmettre une introduction à la thématique et expliciter les points importants à apprendre sur ce sujet. Ensuite crée une liste de  {numberQ} sous-thématique à  pour améliorer la compréhension du sujet {prompt_title}  . Pour chaque sections, veuillez fournir uniquement le titre, sans sous explication de la thématique. Voici la thématique : \n\n1. {prompt_title}"

)

# Prompt templates
question_template = PromptTemplate(
    input_variables = ['list_question', 'number'], 
  template = "À partir de la liste des sous sections suivante : {list_question}, reprenez la sous secftions {number}, reformulez-la  présenter son scéanrio pédagogique avec des objectifs, les minimaux à acquérir. Veuillez énumérer une par une, en sautant une ligne entre chacune, et fournir une explication détaillée pour chaque partie."

)


# Memory 
title_memory = ConversationBufferMemory(input_key='prompt_title', memory_key='chat_history')
listq_memory = ConversationBufferMemory(input_key='list_question', memory_key='list_question_history')

# Llms
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo-16k")
 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='list_question', memory=title_memory)
question_chain = LLMChain(llm=llm, prompt=question_template, verbose=True, output_key='question', memory=listq_memory)

# Show stuff to the screen if there's a prompt
if st.button('Générer le scénario'):

    list_question = title_chain.run(prompt_title=prompt_title, numberQ = numberQ)
    st.info('Proposition de scénario : \n' +  list_question)
    number = 1
   
    for number in range(1, numberQ):
        question = question_chain.run(list_question=list_question, number=number)
        st.info(f'Sous-section : {number} : \n{question}')
        number = number + 1
