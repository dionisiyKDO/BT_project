from flask import flash, jsonify

if __name__ == '__main__':
    training_result = {'loss': 0.9, 'accuracy': 99, 'training_time': 12}

    tmp = jsonify(result=training_result)

    print(tmp)


