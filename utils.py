import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(data):

    if data == 'Diabetes.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/diabetes.csv'
    elif data == 'Breast-Cancer.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/breast-cancer.csv'
    elif data == 'Glass.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/glass_csv.csv'
    elif data == 'Waveform.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/waveform.csv'
    elif data == 'Image.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/image.csv'
    elif data == 'Heart.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/heart.csv'
    elif data == 'Segment.csv':
        dataset = 'https://raw.githubusercontent.com/bullet-ant/Classification-Algorithms-Comparision-App/main/datasets/segment_csv.csv'
    else:
        dataset = None
    return dataset


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_progress_bar(panel):
    # Progress bar logic
    progress_bar = panel.progress(0)
    percent_complete = 1
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
        time.sleep(0.005)

    time.sleep(0.5)
    progress_bar.empty()


def plot_graph(model_report, panel):
    panel.write("")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    d = model_report

    fig = plt.subplots(figsize=(12, 6))

    barWidth = 0.1

    # set heights of bars
    bars1 = d['Perceptron']
    bars2 = d['SVC']
    bars3 = d['GaussianNB']
    bars4 = d['Decision Tree']
    bars5 = d['Gradient Boosting']

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    # Make the plot
    plt.bar(r1, bars1, color='#4c84bc', width=barWidth,
            edgecolor='white', label='Perceptron')
    plt.bar(r2, bars2, color='#c4544c', width=barWidth,
            edgecolor='white', label='SVC')
    plt.bar(r3, bars3, color='#9cbc5c', width=barWidth,
            edgecolor='white', label='GaussianNB')
    plt.bar(r4, bars4, color='#8464a4', width=barWidth,
            edgecolor='white', label='Decision Tree')
    plt.bar(r5, bars5, color='#4cacc4', width=barWidth,
            edgecolor='white', label='Gradient Boosting')

    # Add xticks on the middle of the group bars
    plt.xlabel('Performance Metrics', fontweight='bold')
    # plt.xticks([r + barWidth for r in range(len(bars1))], datasets)
    # plt.xticks(np.arange(4), ['Accuracy', 'Precision', 'Recall', 'F1 SCore'])
    plt.xticks([r + (2 * barWidth) for r in range(len(bars1))],
               ['Accuracy', 'Precision', 'Recall', 'F1 SCore'])
    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0, 1, 1, 0),
               loc="lower left", mode="expand", ncol=5)
    plt.show()
    panel.pyplot()
