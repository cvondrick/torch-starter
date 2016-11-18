require 'torch'
require 'nn'

opt = {
  dataset = 'simple',
  nThreads = 16,
  batchSize = 128,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  cudnn = 1,
  model = 'checkpoints/main/iter100000_net.t7',
  ntest = math.huge,
  randomize = 0,
  cropping = 'center',
  data_root = '/data/vision/torralba/deepscene/places365_standard/',
  data_list = '/data/vision/torralba/commonsense/places-resources/places365/val.txt',
  mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224}
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- load in network
assert(opt.model ~= '', 'no model specified')
print('loading ' .. opt.model)
local net = torch.load(opt.model)
net:evaluate()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

-- ship to GPU
if opt.gpu > 0 then
  input = input:cuda()
  net:cuda()
end

-- eval
local top1 = 0
local top5 = 0
local counter = 0
local maxiter = math.floor(math.min(data:size(), opt.ntest) / opt.batchSize)
for iter = 1, maxiter do
  collectgarbage()
  
  local data_im,data_label = data:getBatch()

  input:copy(data_im)
  local output = net:forward(input)
  local _,preds = output:float():sort(2, true)

  for i=1,opt.batchSize do
    local rank = torch.eq(preds[i], data_label[i]):nonzero()[1][1]
    if rank == 1 then
      top1 = top1 + 1
    end
    if rank <= 5 then
      top5 = top5 + 1
    end
  end

  counter = counter + opt.batchSize
  
  print(('%s: Eval [%8d / %8d]:\t Top1: %.4f  Top5: %.4f'):format(
    opt.model, iter, maxiter,
    top1/counter, top5/counter))
end
