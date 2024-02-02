import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TicTacToeGameModel(torch.nn.Module):

    def __init__(self):
        super(TicTacToeGameModel, self).__init__()

        self.linear1 = nn.Linear(9*2, 256) # Track X's, O's and blanks -- 3 boards
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(256, 9)
        #self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        #x = self.softmax(x)
        return x

## Define some expected moves
move1 = """
---
---
---

X--
---
---
"""

move2 = """
XOX
X-O
---

---
---
O--
"""

moves = [move1, move2]

def parse_move(move):
    compress = move.replace('\n', '')
    before = compress[:9]
    after = compress[9:18]
    return before, after

def convert_board_to_tensor(board):
    tensor = torch.FloatTensor([0] * 9*2) # 9 per board, 2x for each player
    for i, space in enumerate(board):
        if space == 'X':
            tensor[i] = 1.0;
        elif space == 'O':
            tensor[i + 9] = 1.0;
        elif space == '-':
            pass
        else:
            raise ValueException(f'Unexpected board character {space} at {i} for board {board}')
    return tensor

# Where the expected move is
def convert_move_to_tensor(board):
    tensor = torch.FloatTensor([0] * 9)
    for i, space in enumerate(board):
        if space == 'X' or space == 'O':
            tensor[i] = 1.0;
        elif space == '-':
            pass
        else:
            raise ValueException(f'Unexpected board character {space} at {i} for board {board}')
    return tensor


if __name__ == '__main__':

    before, after = parse_move(move2)
    print(before, after)

    print("before:")
    print(convert_board_to_tensor(before))
    print("after:")
    print(convert_board_to_tensor(after))

    model = TicTacToeGameModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    def train_model(num_epochs, learning_rate):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, move in enumerate(moves, 0):
                before, after = parse_move(move)
                inputs = convert_board_to_tensor(before)
                expected_move = convert_move_to_tensor(after)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, expected_move)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%s, %d, %5d] loss: %.3f' %
                          (learning_rate, epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0


    # Follow the LeNet-5 manuscript training epochs and learning rates
    train_model(100, 0.0005)
    train_model(100, 0.0001)

print('Finished Training')


# Show the results
with torch.no_grad():
    correct = 0
    total = 0
    wrong_prediction_count = 0
    correct_prediction_count = 0
    for i, move in enumerate(moves, 0):
        before, after = parse_move(move)
        inputs = convert_board_to_tensor(before)
        output = model(inputs)
        expected_move = convert_move_to_tensor(after)

        prediction = (output.max() == output).nonzero(as_tuple=True)[0].item()
        print(prediction)
        # Save images for some of the incorrect predictions
        if not expected_move[prediction]:
            if wrong_prediction_count < 10:
                wrong_prediction_count += 1
                print(f'Wrong. Predicted {prediction} but it was {expected_move}. Logits: {output}')
        # Save image for some of the correct predictions
        else:
            if correct_prediction_count < 10:
                correct_prediction_count += 1
                print(f'Right. Predicted {prediction} and labeled {expected_move}. Logits: {output}')

