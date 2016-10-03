require 'image'

local cmd = torch.CmdLine()

cmd:option('-scale', 1, 'Factor to scale by.')
cmd:option('-input', '', 'Paths of image to stylize.')
cmd:option('-output', '', 'Path to save stylized image.')

local params = cmd:parse(arg)

local files = {}
for file in paths.iterfiles(params.input) do table.insert(files, file) end
table.sort(files)

for _,file in pairs(files) do
  print(file)
  local source = paths.concat(params.input, file)
  local dest = paths.concat(params.output, file)

  local source_img = image.load(source, 3):float()
  source_img = image.scale(source_img, source_img:size(3)*params.scale, source_img:size(2)*params.scale)

  image.save(dest, source_img)
end
