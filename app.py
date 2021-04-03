import streamlit as st
import utils
import pandas as pd
import time
import models as models
from classifiers import BatchClassifier
from sklearn.model_selection import train_test_split

#  Execution STARTS from here


def main():

    # Headline for all Pages
    st.write(''' # Machine Learning: A comparison of classification algorithms
    
    ---  
    
    ''')

    # Page 1: Instructions on main_panel if "Show instructions" button is selected from sidebar

    instructions = st.markdown("""

        Jumpstart to find the best class of algorithm for your data:

        >   1. 👈 Select **Run the app** from the sidebar *(click on **>** if closed)* 
        2. Select a dataset to get started
        3. Click "📊 Visualize Dataset" to analyze the dataset
        4. Click ":rocket: Build Models" to start comparision among different models
        5. Select a model for hyperparameter tuning and click ":rocket: Tune Model"
        6. Find the best model & do your magic! :sparkles:
        
        ---
        """)

    # Sidebar structure STARTS here

    st.sidebar.write('''
        ## Welcome!
        ''')

    # Navigation selectbox

    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Run the app"])

    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "Run the app":
        instructions.empty()
        run_the_app()


# Page 2 : Structure when "Run the App" is selected in sidebar
def run_the_app():

    # Sidebar container below the Navigation Selectbox
    sidebar = st.sidebar.beta_container()

    # Main Panel section below Heading
    main_panel = st.beta_container()

    ############################
    # Button 1
    ############################

    # Dataset selector
    sidebar.markdown('#### **1. Select a dataset**')
    dataset_selector = sidebar.selectbox('', ('Click here...', 'Diabetes.csv', 'Breast-Cancer.csv',
                                              'Waveform.csv', 'Image.csv', 'Segment.csv', 'Glass.csv', 'Heart.csv'))
    sidebar.write("")

    sidebar.markdown('#### **Or Upload your .csv file**')
    uploaded_dataset = sidebar.file_uploader(
        "Upload your dataset", type=['.csv'])

    # Button to show more insights on dataset
    visualizer_button = sidebar.empty()

    # Sidebar container to show parameters selection options
    parameters_sidebar = st.sidebar.beta_container()

    # Sidebar button to fire "Model Building Process"
    model_sidebar = st.sidebar.beta_container()
    model_runner = st.sidebar.empty()

    if dataset_selector == 'Click here...' and uploaded_dataset is None:  # i.e. No dataset is selected
        main_panel.info('Awaiting for user to select a dataset...')
    else:

        # ======================= SIDEBAR ==================================

        ############################
        # Button 2
        ############################

        visualize_status = visualizer_button.button(
            f'📊 Visualize {dataset_selector}\n')
        # Adding parameters selection in container named "parameter_sidebar"
        with parameters_sidebar.markdown('#### **2. Set Parameters**'):

            # Split Slider Selector
            split_size = parameters_sidebar.slider(
                'Data split ratio (% for Training Set)', 10, 90, 80, 5)
            seed_number = parameters_sidebar.slider(
                'Set the random seed number', 1, 100, 42, 1)

        ############################
        # Button 3
        ############################

        model_sidebar.markdown('#### **3. Analyze Model**')
        model_sidebar.write("")
        model_runner_status = model_runner.button(f'🚀 Build Models')

        # ========================== MAIN PANEL =============================

        if uploaded_dataset is not None:
            dataset = uploaded_dataset
        else:
            dataset = utils.get_dataset(dataset_selector)

        if not model_runner_status:
            # Fetch .csv of selected dataset

            with main_panel:

                df = pd.read_csv(dataset)

                if dataset_selector != 'Click here...' or uploaded_dataset is not None:
                    if visualize_status is True:
                        if uploaded_dataset is not None:
                            visualizer(df, uploaded_dataset.name)
                        else:
                            visualizer(df, dataset_selector)
                    else:
                        message = main_panel.empty()
                        if uploaded_dataset is not None:
                            message.info(
                                f'Fetching {uploaded_dataset.name}')
                        else:
                            message.info(
                                f'Fetching {dataset_selector}')

                        # Loads a progress bar
                        utils.get_progress_bar(main_panel)

                        if uploaded_dataset is not None:
                            message.success(
                                f'Successfully loaded {uploaded_dataset.name}')
                        else:
                            message.success(
                                f'Successfully loaded {dataset_selector}')

                        time.sleep(1)

                        message.empty()
                        # Write dataframe in main panel
                        if uploaded_dataset:
                            main_panel.write(
                                f'''#### Loaded dataset: {uploaded_dataset.name}''')
                        else:
                            main_panel.write(
                                f'''#### Loaded dataset: {dataset_selector}''')

                        main_panel.write("")
                        main_panel.write(df)

    # =========================== MAIN PANEL ENDS ===============================================

        else:
            # Model Graphs loading starts heree

            utils.get_progress_bar(main_panel)

            if uploaded_dataset is not None:
                models_report = build_on_uploaded_dataset(
                    uploaded_dataset)
                main_panel.write(models_report)
                # models_names = ['Perceptron', 'SVC', 'Gaussian NB',
                #                 'Decision Tree', 'Gradient Boosting']
                # models_report.insert(0, 'models', models_names)
                models_report_dict = models_report.to_dict(orient='split')
                models_report_metrics = {}
                models_report_metrics['Perceptron'] = models_report_dict['data'][0][1:5]
                models_report_metrics['SVC'] = models_report_dict['data'][1][1:5]
                models_report_metrics['GaussianNB'] = models_report_dict['data'][2][1:5]
                models_report_metrics['Decision Tree'] = models_report_dict['data'][3][1:5]
                models_report_metrics['Gradient Boosting'] = models_report_dict['data'][4][1:5]
                # print(models_report_metrics)
                utils.plot_graph(models_report_metrics, main_panel)

            else:
                classification_reports, model_report = models.builder(
                    dataset, (1 - (split_size / 100)))

                report_df = pd.DataFrame.from_dict(classification_reports)
                main_panel.write(report_df)

                utils.plot_graph(model_report, main_panel)


# Visualizer for Dataset

def visualizer(df, dataset_selector):
    visualize = st.beta_container()
    with visualize:
        visualize.markdown(
            f"### **1. Dataset: {dataset_selector}** \n #### 1.1. Glimplse of dataset \n")
        visualize.write(df.head(10))

        # Using all column except for the last column as X

        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        visualize.markdown('#### 1.2. Dataset dimension')
        cols = visualize.beta_columns(2)
        cols[0].write('X')
        cols[0].info('{} rows, {} attributes'.format(
            X.shape[0], X.shape[1]))
        cols[1].write('Y')
        cols[1].info('{} responses'.format(Y.shape[0]))

        visualize.markdown('#### 1.3. Variable details:\n')
        visualize.markdown('**X variables** (first 20 are shown)')

        attributes = list(X.columns[:20])
        visualize.code('{}'.format(attributes))

        visualize.markdown('**Y variable**')
        visualize.code('{}'.format(Y.name))

# Logic for handling uploaded data


def build_on_uploaded_dataset(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.5)
    clf = BatchClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)

    del models['Balanced Accuracy']
    del models['ROC AUC']
    # rename province to state
    models.rename(columns={'Accuracy': 'accuracy', 'Precision': 'precision',
                           'F1 Score': 'f1', 'Recall': 'recall', 'Time Taken': 'training_time', 'Model': 'model'}, inplace=True)

    return models


# Driver Function


if __name__ == '__main__':
    st.set_page_config(
        page_title='Machine Learning: A comparison of classification algorithms')
    main()
