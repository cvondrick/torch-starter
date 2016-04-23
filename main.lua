require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'simple',   -- indicates what dataset load to use (in data.lua)
  nThreads = 16,        -- how many threads to pre-fetch data
  batchSize = 256,      -- self-explanatory
  loadSize = 256,       -- when loading images, resize first to this size
  fineSize = 224,       -- crop this size from the loaded image 
  nClasses = 401,       -- number of category
  lr = 0.001,           -- learning rate
  lr_decay = 1,         -- how often to decay learning rate (in epoch's)
  beta1 = 0.5,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  niter = 100,          -- number of iterations through dataset
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  name = 'net1',        -- the name of the experiment
  randomize = true,     -- whether to shuffle the data file or not
  display_port = 8000,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  data_root = '/data/vision/torralba/commonsense/places-resources/flat/',
  data_list = '/data/vision/torralba/commonsense/places-resources/flat/train_class.txt'
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
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
if opt.finetune == '' then -- build network from scratch
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

  -- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init) -- loop over all layers, applying weights_init

else -- load in existing network
  print('loading ' .. opt.finetune)
  net = torch.load(opt.finetune)
end

print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local err

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  net:cuda()
  criterion:cuda()
end

-- conver to cudnn if needed
-- if this errors on you, you can disable, but will be slightly slower
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end

-- estimate the channel-wise mean
-- if you don't want to do this everytime, set meanIter=0 and
-- hard code the mean in the next line:
local mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224}
if opt.meanIter > 0 then
  mean = {0,0,0}
  for i=1,opt.meanIter do
    print('Estimating mean: ' .. i .. ' / ' .. opt.meanIter)
    data_im = data:getBatch()
    for c=1,3 do
      mean[c] = mean[c] + data_im:select(2,c):mean() 
    end
  end
  for c=1,3 do
    mean[c] = mean[c] / opt.meanIter
  end
end
print('Mean is:')
print(mean)
net.mean = mean

-- subtracts the mean in place
local function subtractMean(images)
  for c=1,3 do images[{ {}, c, {}, {} }]:add(-mean[c]) end
end

-- get a vector of parameters
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
  subtractMean(data_im)
  data_tm:stop()

  -- ship data to GPU
  input:copy(data_im:squeeze())
  label:copy(data_label)
  
  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  net:backward(input, df_do)
  
  -- return gradients
  return err, gradParameters
end

local counter = 0
local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train main loop
for epoch = 1,opt.niter do -- for each epoch
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do -- for each mini-batch
    collectgarbage() -- necessary sometimes
    
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
  
  -- save checkpoint
  -- :clearState() compacts the model so it takes less space on disk
  print('Saving ' .. opt.name .. '_' .. epoch .. '_net.t7')
  paths.mkdir('checkpoints')
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net.t7', net:clearState())
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_optim.t7', optimState)
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_history.t7', history)

  -- decay the learning rate, if requested
  if opt.lr_decay > 0 and epoch % opt.lr_decay == 0 then
    opt.lr = opt.lr / 10
    print('Decreasing learning rate to ' .. opt.lr)

    -- create new optimState to reset momentum
    optimState = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
    }
  end
end
