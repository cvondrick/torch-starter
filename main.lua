require 'torch'
require 'nn'
require 'optim'

opt = {
  dataset = 'simple',
  nThreads = 16,
  batchSize = 100,
  loadSize = 256,
  fineSize = 224,
  nClasses = 401,
  lr = 0.001,
  beta1 = 0.5,
  niter = 100,
  ntrain = math.huge,
  gpu = 1,
  finetune = '',
  name = 'net1',
  randomize = true,
  display_port = 8000, 
  display_id = 1, 
  data_root = '/data/vision/torralba/commonsense/places-resources/flat/',
  data_list = '/data/vision/torralba/commonsense/places-resources/flat/train_class.txt'
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

-- define the model
local net
if opt.finetune == '' then
  net = nn.Sequential()
  net:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
  net:add(nn.SpatialBatchNormalization(64,1e-3))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
  net:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
  net:add(nn.SpatialBatchNormalization(192,1e-3))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
  net:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
  net:add(nn.SpatialBatchNormalization(384,1e-3))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
  net:add(nn.SpatialBatchNormalization(256,1e-3))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
  net:add(nn.SpatialBatchNormalization(256,1e-3))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
  net:add(nn.View(256*6*6))
  net:add(nn.Dropout(0.5))
  net:add(nn.Linear(256*6*6, 4096))
  net:add(nn.BatchNormalization(4096, 1e-3))
  net:add(nn.ReLU())
  net:add(nn.Dropout(0.5))
  net:add(nn.Linear(4096, 4096))
  net:add(nn.BatchNormalization(4096, 1e-3))
  net:add(nn.ReLU())
  net:add(nn.Linear(4096, opt.nClasses))

  ---- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init)
else
  print('loading ' .. opt.fineTune)
  net = torch.load(opt.finetune)
end
print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local err

local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship to GPU
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  net:cuda()
  criterion:cuda()
end

-- get parameters
local parameters, gradParameters = net:getParameters()

-- show graphics
disp = require 'display'
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im,data_label
local fx = function(x)
  gradParameters:zero()
  
  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im,data_label = data:getBatch()
  data_tm:stop()

  -- ship data to GPU
  input:copy(data_im:squeeze())
  label:copy(data_label)
  
  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  net:backward(input, df_do)
  
  -- get parameters
  return err, gradParameters
end

local counter = 0
local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train
for epoch = 1,opt.niter do
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    collectgarbage()
    
    tm:reset()

    -- do one iteration
    optim.adam(fx, parameters, optimState)
    
    -- logging
    if counter % 10 == 0 then
      table.insert(history, {counter, err})
      disp.image(data_im, {win=opt.display_id, title=(opt.name .. ' batch')})
      disp.plot(history, {win=opt.display_id+1, title=opt.name, labels = {"iteration", "err"}})
      disp.image(net.modules[1].weight, {win=opt.display_id+2, title=(opt.name .. ' conv1')})
    end
    counter = counter + 1
    
    print(('%s: Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
              .. '  Err: %.4f'):format(
            opt.name, epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            err and err or -1))
  end
  
  paths.mkdir('checkpoints')
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net.t7', net:clearState())
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_optim.t7', optimState)
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_history.t7', history)
end
