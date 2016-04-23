require 'torch'
require 'nn'

opt = {
  dataset = 'simple',
  nThreads = 16,
  batchSize = 128,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  name = 'net1',
  epoch = 1,
  ntest = math.huge,
  randomize = false,
  data_root = '/data/vision/torralba/commonsense/places-resources/flat/',
  data_list = '/data/vision/torralba/commonsense/places-resources/flat/val_class_shuf.txt'
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

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- load in network
print('loading ' .. opt.name .. ' epoch=' .. opt.epoch)
local net = torch.load('checkpoints/' .. opt.name .. '_' .. opt.epoch .. '_net.t7')
net:evaluate()

-- subtracts the mean in place
local function subtractMean(images)
  for c=1,3 do images[{ {}, c, {}, {} }]:add(-net.mean[c]) end
end

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
  subtractMean(data_im)

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
    opt.name, iter, maxiter,
    top1/counter, top5/counter))
end
