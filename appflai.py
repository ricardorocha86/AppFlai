import streamlit as st
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'App FLAI - Powered by Streamlit', 
				   page_icon = 'iconeflai.png' ,
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-para-previsao-de-salario')

@st.cache
def ler_dados():
	dados = pd.read_csv('prof-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

st.image('bannerflai.jpg', use_column_width = 'always')

st.sidebar.write('''
# :sparkles: **[App FLAI](https://www.flai.com.br/)**
***[Powered by Streamlit](https://streamlit.io/)***

---

	''')
 
opcoes = paginas = ['Home', 'Análise de Dados', 'Dashboard', 'Modelo de Proposta de Salário', 'Streamlit Widgets', 'Sobre', 'Código']
pagina = st.sidebar.radio('Selecione uma página:', paginas)
#st.sidebar.markdown('---')

if pagina == 'Home':
	

	st.write("""
	# :sparkles: Bem-vindo ao App FLAI
	***Powered by Streamlit***
	
	---

	Nesse Web App vamos fazer análises de dados rápidas, dashboards e deploy de um modelo para estimar salários de profissionais da área de dados.

	### Funcionalidades no momento
	
	:ballot_box_with_check:  Página Inicial: Home
	
	:ballot_box_with_check:  Sub-página para Análise de Dados

	:ballot_box_with_check:  Sub-página para Dashboards

	:ballot_box_with_check:  Sub-página para Deploy do Modelo de Estimação de Salário de Profissionais de Dados no Brasil
	
	:black_square_button:  Deploy em lote (de vários pessoas ao mesmo tempo, a partir de um arquivo)
	
	:ballot_box_with_check:  Página Sobre 

	Os modelos desse web-app foram desenvolvidos utilizando o conjunto de 
	dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/datahackers/pesquisa-data-hackers-2019).
	
	Os arquivos para gerar esse aplicativo estão nesse [link](https://github.com/gitflai/imersao). 
	
	Os modelos são desenvolvidos e analisados utilizando a biblioteca [PyCaret](https://pycaret.org/).
	
	Caso encontre algum erro/bug, por favor, não hesite em entrar em contato! :poop:
	
	Para mais informações sobre o Streamlit, consulte o [site oficial](https://www.streamlit.io/) ou a sua [documentação](https://docs.streamlit.io/_/downloads/en/latest/pdf/).
	
	[Lista de emojis para markdown](https://gist.github.com/rxaviers/7360908)
 	
 

	""")





if pagina == 'Análise de Dados': 
	st.markdown('''
		## **Análise de Dados**
		**Utilize essa página para explorar as variáveis do conjunto de dados utilizado.**
		''')
	variaveis = dados.columns.to_list()

	st.markdown('---')
	st.markdown('### Uma amostra dos dados:')
	st.write(dados.sample(10))

	st.markdown('---')
	st.markdown('### Algumas descritivas dos dados:')
	st.table(dados.describe())

	st.markdown('---')
	st.markdown('### Gráficos de Contagem:')
	var = st.selectbox('Selecione uma variável:', variaveis) 
	g1  = dados[var].value_counts().plot(kind = 'barh', title = 'Contagem {}'.format(var)) 
	st.pyplot(g1.figure)
 
	st.markdown('---')
	st.markdown('### Gráficos do Salário por Variável:')
	lvar2 = variaveis.copy()
	lvar2.pop(0) 
	var1 = st.selectbox('Selecione:', lvar2)
	g2 = dados['Salário'].groupby(dados[var1]).mean().plot(kind = 'barh', title = 'Salário por {}'.format(var1))
	st.pyplot(g2.figure)


	st.markdown('---')
	st.markdown('### Gráficos do Salário em relação a duas Variáveis:')
	v1 = st.selectbox('Selecione uma variável:', lvar2)
	v2 = st.selectbox('Selecione outra variável:', lvar2)
	titulo = 'Salário por {} e {}'.format(v1, v2)
	g3 = dados.groupby([v1, v2]).mean()['Salário'].unstack().plot(kind = 'barh', title = titulo)
	st.pyplot(g3.figure)


if pagina == 'Dashboard': 

	st.sidebar.markdown('### **Menu Complementar**')
	vrs = dados['Profissão'].unique().tolist()
	vrs.remove('Outras')
	prof = st.sidebar.radio('Profissão', vrs , index = 3)
	dados0 = dados[dados['Profissão'] == prof]
	n = dados0.shape[0]
	s = dados0['Salário'].mean()

	st.markdown('# Dashboard dos **{}**'.format(prof))

	st.markdown('---')
	col1, col2 = st.beta_columns((1, 2))
	col1.markdown('### Amostra: **{}**'.format(n))
	col2.markdown('### Salário: **R${:.2f}**'.format(s)) 

	st.markdown('---')
	col1, col2 = st.beta_columns((1, 2))

	d1 = dados0['Idade'].value_counts().plot(kind = 'barh')
	col1.pyplot(d1.figure, clear_figure = True)

	d2 = dados0['Linguagem Python'].value_counts().plot(kind = 'pie', title ='Python') 
	col1.pyplot(d2.figure, clear_figure = True)

	titulo = 'Salário por Idade e Tamanho da Empresa'
	d3 = dados0.groupby(['Idade', 'Tamanho da Empresa']).mean()['Salário'].unstack().plot(kind = 'barh', title = titulo)
	col2.pyplot(d3.figure, clear_figure = True)

	st.markdown('---')


if pagina == 'Modelo de Proposta de Salário': 
	st.markdown('---')
	st.markdown('## **Modelo para Estimar o Salário de Profissionais da área de Dados**')
	st.markdown('Utilize as variáveis abaixo para utilizar o modelo de previsão de salários desenvolvido [aqui]().')
	st.markdown('---')

	col1, col2, col3 = st.beta_columns(3)

	x1 = col1.radio('Idade', dados['Idade'].unique().tolist() )
	x2 = col1.radio('Profissão', dados['Profissão'].unique().tolist())
	x3 = col1.radio('Tamanho da Empresa', dados['Tamanho da Empresa'].unique().tolist())
	x4 = col1.radio('Cargo de Gestão', dados['Cargo de Gestão'].unique().tolist())
	x5 = col3.selectbox('Experiência em DS', dados['Experiência em DS'].unique().tolist()) 
	x6 = col2.radio('Tipo de Trabalho', dados['Tipo de Trabalho'].unique().tolist() )
	x7 = col2.radio('Escolaridade', dados['Escolaridade'].unique().tolist())
	x8 = col3.selectbox('Área de Formação', dados['Área de Formação'].unique().tolist())
	x9 = col3.selectbox('Setor de Mercado', dados['Setor de Mercado'].unique().tolist())
	x10 = 1
	x11 = col2.radio('Estado', dados['Estado'].unique().tolist()) 
	x12 = col3.radio('Linguagem Python', dados['Linguagem Python'].unique().tolist()) 
	x13 = col3.radio('Linguagem R', dados['Linguagem R'].unique().tolist()) 
	x14 = col3.radio('Linguagem SQL', dados['Linguagem SQL'].unique().tolist()) 
	 

	dicionario  =  {'Idade': [x1],
				'Profissão': [x2],
				'Tamanho da Empresa': [x3],
				'Cargo de Gestão': [x4],
				'Experiência em DS': [x5],
				'Tipo de Trabalho': [x6],
				'Escolaridade': [x7],
				'Área de Formação': [x8],
				'Setor de Mercado': [x9],
				'Brasil': [x10],
				'Estado': [x11],		
				'Linguagem Python': [x12],
				'Linguagem R': [x13],
				'Linguagem SQL': [x14]}

	dados = pd.DataFrame(dicionario)  

	st.markdown('---') 
	st.markdown('## **Quando terminar de preencher as informações da pessoa, clique no botão abaixo para estimar o salário de tal profissional**') 


	if st.button('EXECUTAR O MODELO'):
		saida = float(predict_model(modelo, dados)['Label']) 
		st.markdown('## Salário estimado de **R$ {:.2f}**'.format(saida))









if pagina == 'Streamlit Widgets':
	# col1, col2 = st.beta_columns(2) 
	st.markdown('---')

	st.markdown('### **Botões**')
	st.markdown('Guardam valores **True** ou **False**')
	st.code("st.button(label = '-> Clique aqui! <-', help = 'É só clicar ali')")
	st.button(label = '-> Clique aqui! <-', help = 'É só clicar ali')

	st.markdown('---')

	st.markdown('### **Caixa de Selecionar**')
	st.markdown('Guardam valores **True** ou **False**')
	st.code("st.checkbox('Clique para me selecionar', help = 'Clique e desclique quando quiser')")
	st.checkbox('Clique para me selecionar', help = 'Clique e desclique quando quiser')

	st.markdown('---')

	st.markdown('### **Botões de Rádio**')
	st.markdown('Guarda o item do botão selecionado')
	st.code("st.radio('Botões de Rádio', options = [100, 'Python', print, [1, 2, 3]], index = 1, help = 'Ajuda')")
	st.radio('Botões de Rádio', options = [100, 'Python', print, [1, 2, 3]], index = 1, help = 'Ajuda')

	st.markdown('---')

	st.markdown('### **Caixas de Seleção**')
	st.markdown('Guarda o item da caixa selecionado')
	st.code("st.selectbox('Clique no item que deseja', options = ['azul', 'roxo', 'verde'], index = 2)")
	st.selectbox('Clique no item que deseja', options = ['azul', 'roxo', 'verde'], index = 2)

	st.markdown('---')

	st.markdown('### **Caixas de Seleção Múltipla**')
	st.markdown('Guarda a lista de itens selecionados')
	st.code("st.multiselect('Selecione quantas opções desejar', options = ['A', 'B', 'C', 'D', 'E'])")
	st.multiselect('Selecione quantas opções desejar', options = ['A', 'B', 'C', 'D', 'E'])
	
	st.markdown('---')

	st.markdown('### **Slider**')
	st.markdown('Guarda o número selecionado')
	st.code("st.slider('Entrada numérica', min_value = 1, max_value = 25, value = 7, step = 2)	")
	st.slider('Entrada numérica', min_value = 1, max_value = 25, value = 7, step = 2)	
	
	st.markdown('---')

	st.select_slider('Slide to select', options=[1,'2'])
	st.markdown('---')

	st.text_input('Entrada de Texto')
	st.markdown('---')

	st.number_input('Entre com um número')
	st.markdown('---')

	st.text_area('Entre com um Textão')
	st.markdown('---')

	st.date_input('Entre com uma data')
	st.markdown('---')

	st.time_input('Entre com um horário')
	st.markdown('---')

	st.file_uploader('Suba um arquivo do seu computador')
	st.markdown('---')

	st.color_picker('Escolha uma cor')
	st.markdown('---')



if pagina == 'Sobre':

	st.markdown(""" 

	## **Sobre**

	Nesse Web App mostramos o poder do streamlit para construir soluções fáceis, rápidas e que permitem uma usabilidade bastante ampla.')
		
	Pare por um momento e imagine o universo de possibilidades que temos ao combinar\
		todos os recursos do streamlit com o que já temos no Python.
		
	Esse tipo de web-app é perfeito para quando se quer entregar uma solução rápida\
		e/ou criar um ambiente de testes mais eficiente.

	Não deixe de explorar os recursos do streamlit. Aprenda, crie, desenvolva. \
		Faça o que ninguém fez ainda. Vá além. 

	*#itstimetoflai* :rocket:

	---
	
             """) 

	if st.button('Comemorar'):
		st.balloons()



if pagina == 'Código':

 	st.code('def aux(x): ...', language='python')


st.sidebar.markdown('---')
st.sidebar.image('logoflai.png', width = 90)



