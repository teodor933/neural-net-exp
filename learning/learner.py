

class Learner:
    def __init__(self, model, loss_fn, optimiser, batch_size=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.batch_size = batch_size

    def learn_step(self, input_batch, output_batch):
        pred, _ = self.model.forward(input_batch)

        loss = self.loss_fn.execute(pred, output_batch)

        d_loss = self.loss_fn.derivative(pred, output_batch)
        self.model.backpropagate(d_loss, self.optimiser) # iterate network

        return loss # other use cases

    def learn(self, input_batch, output_batch, epochs,
              verbose=True):
        for epoch in range(epochs):
            loss = self.learn_step(input_batch, output_batch)

            if verbose and epoch % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

        return self