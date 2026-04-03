% Dataset: [Attendance Assignment Test Performance]
data = [
90 85 88 90;
80 75 78 82;
60 55 58 60;
70 65 68 70;
95 90 92 94;
85 80 82 85;
];

% Generate initial FIS
fis = genfis1(data, 3);

% Train ANFIS model
[trainedFis, error] = anfis(data, fis, 50);

% Test input
testInput = [85 80 82];
result = evalfis(testInput, trainedFis);

disp('Predicted Performance:')
disp(result)

% Plot training error
figure;
plot(error)
title('Training Error')
xlabel('Epochs')
ylabel('Error')