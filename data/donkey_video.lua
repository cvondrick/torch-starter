--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

require 'torch'
require 'image'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')

-- this function reads in the data files
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  assert(self.frameSize > 0, "must specify a number of frames to load")

  -- how many zeros to pad in the frames filename
  if self.filenamePad == nil then
    self.filenamePad = 8
  end

  -- read text file consisting of frame directories and counts of frames
  self.data = tds.Vec()
  self.counts = tds.Vec()
  print('reading ' .. args.data_list)
  for line in io.lines(args.data_list) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    split[2] = tonumber(split[2])
    if split[2] ~= nil and split[2] > self.frameSize then
      self.data:insert(split[1])
      self.counts:insert(split[2])
    end
  end

  print('found ' .. #self.data .. ' videos')

end

function dataset:size()
  return #self.data
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, extraTable)
   local data, scalarLabels, labels
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)
   data = torch.Tensor(quantity, 3, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end
   return data, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local extraTable = {}
   for i=1,quantity do
      local idx = torch.random(1, #self.data)
      local data_path = self.data_root .. '/' .. self.data[idx]
      local data_count = self.counts[idx]

      local out = self:trainHook(data_path, data_count) 
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])
   end
   return self:tableToOutput(dataTable,extraTable)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
   assert(start_idx)
   assert(stop_idx)
   assert(start_idx<stop_idx)
   assert(start_idx<=#self.data)
   local dataTable = {}
   local extraTable = {}
   for idx=start_idx,stop_idx do
      if idx > #self.data then
        break
      end
      local data_path = self.data_root .. '/' .. self.data[idx]
      local data_count = self.label[idx]

      local out = self:trainHook(data_path, data_count) 
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])
   end
   return self:tableToOutput(dataTable,extraTable)
end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(path, count)
  collectgarbage()
  local t1 = torch.random(1, count - self.frameSize) 

  local oW = self.fineSize
  local oH = self.fineSize 
  local h1
  local w1

  local out = torch.zeros(3, self.frameSize, oW, oH)
  
  for fr=1,self.frameSize do
    local filename = path .. '/' .. string.format("/%0" .. self.filenamePad .. "d.jpg", (fr-1) + t1)
    local input = self:loadImage(filename)
    local iW = input:size(3)
    local iH = input:size(2)

    if fr == 1 then
      -- do random crop
      if self.cropping == 'random' then
        h1 = math.ceil(torch.uniform(1e-2, iH-oH))
        w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      elseif self.cropping == 'center' then
        h1 = math.ceil((iW-oW)/2)
        w1 = math.ceil((iH-oH)/2)
      else
        assert(false, 'unknown mode ' .. self.cropping)
      end
    end

    local input2 = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    assert(input2:size(2) == oW)
    assert(input2:size(3) == oH)

    out[{ {}, fr, {}, {} }]:copy(input2)
  end

  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- subtract mean
  for c=1,3 do
    out[{ c, {}, {} }]:add(-self.mean[c])
  end

  return out
end

-- reads an image disk
-- if it fails to read the image, it will use a blank image
-- and write to stdout about the failure
-- this means the program will not crash if there is an occassional bad apple
function dataset:loadImage(path)
  local ok,input = pcall(image.load, path, 3, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     input = torch.zeros(3, opt.loadSize, opt.loadSize) 
  else
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
        input = image.scale(input, opt.loadSize, opt.loadSize * iH / iW)
    else
        input = image.scale(input, opt.loadSize * iW / iH, opt.loadSize) 
    end
  end

  return input
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt)
