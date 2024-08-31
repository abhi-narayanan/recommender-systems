from flask import Flask, render_template, request
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pkl', 'rb'))
pivot_table = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
sim_score = pickle.load(open('similarity_scores.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html",
                           book_name = list(popular_df['Book-Title'].values),
                           author = list(popular_df['Book-Author'].values),
                           image = list(popular_df['Image-URL-M'].values),
                           votes = list(popular_df['num_ratings'].values),
                           rating = list(popular_df['avg_ratings'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template("recommend.html")

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    # Index of book_name
    index = np.where(pivot_table.index == user_input)[0][0]
    distances = sim_score[index]
    similar_books = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pivot_table.index[i[0]]]
        temp_df = temp_df.drop_duplicates('Book-Title')
        item.extend(list(temp_df['Book-Title'].values))
        item.extend(list(temp_df['Book-Author'].values))
        item.extend(list(temp_df['Image-URL-M'].values))

        data.append(item)

    # print(data)

    return render_template('recommend.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)