# functions/add.py
from torch.autograd import Function
from . import _roi_crop as roi_crop


class RoICropFunction(Function):
  layout = "BHWD"

  @staticmethod
  def forward(ctx, input1, input2):

    ctx.save_for_backward(input1, input2)
    output = input2.new(input2.size()[0],
                        input1.size()[1],
                        input2.size()[1],
                        input2.size()[2]).zero_()
    assert (output.get_device() == input1.get_device(),
            "output and input1 must on the same device")
    assert (output.get_device() == input2.get_device(),
            "output and input2 must on the same device")

    if RoICropFunction.layout == "BHWD":
      roi_crop.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output)
    elif RoICropFunction.layout == "BCHW":
      roi_crop.BilinearSamplerBCHW_updateOutput(input1, input2, output)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    input1, input2 = ctx.saved_tensors
    grad_input1 = input1.new(input1.size()).zero_()
    grad_input2 = input2.new(input2.size()).zero_()

    if RoICropFunction.layout == "BHWD":
      roi_crop.BilinearSamplerBHWD_updateGradInput_cuda(input1,
                                                        input2,
                                                        grad_input1,
                                                        grad_input2,
                                                        grad_output)
    elif RoICropFunction.layout == "BCHW":

      roi_crop.BilinearSamplerBCHW_updateGradInput(input1,
                                                   input2,
                                                   grad_input1,
                                                   grad_input2,
                                                   grad_output)
    return grad_input1, grad_input2
