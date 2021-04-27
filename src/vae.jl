using BSON
using CUDA
using DrWatson: struct2dict
using Flux
using Flux: @functor, chunk
#using Flux.Losses: logitbinarycrossentropy
using Flux.Losses: binarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using PyPlot
#device!(1)
import Statistics

#include("ex_dan.jl")

#filename="trainingdata.jld2"
#variablename="allloghycos"

function loaddata(filename, variablename)
	data = load(filename, variablename)
	numpixels = size(data, 2)
	xtrn = Array{Float32}(undef, numpixels, numpixels, 1, div(8 * size(data, 1), 10))
	ytrn = ones(Float32, div(8 * size(data, 1), 10))
        xtst = Array{Float32}(undef, numpixels, numpixels, 1, size(data, 1) - div(8 * size(data, 1), 10))
	ytst = ones(Float32, size(data, 1) - div(8 * size(data, 1), 10))
	lowend = minimum(data)
	highend = maximum(data)
	for i = 1:size(xtrn, 4)
        	xtrn[:, :, 1, i] = (data[i, :, :] .- lowend) ./ (highend - lowend)
    	end
    	for i = 1:size(xtst, 4)
        	xtst[:, :, 1, i] = (data[i + div(8 * size(data, 1), 10), :, :] .- lowend) ./ (highend - lowend)
    	end
    	return xtrn, ytrn, xtst, ytst, highend, lowend
end

# load perm images and return loader
function get_data(batch_size,filename,variablename)
    xtrn, ytrn, xtst, ytst, highend, lowend  = loaddata(filename, variablename)
    xtrn = reshape(xtrn, 100^2, :)
    xtrain = xtrn
    ytrain = ytrn
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

function get_data2(batch_size,filename,variablename)
    xtrn, ytrn, xtst, ytst, highend, lowend  = loaddata(filename, variablename)
    xtst = reshape(xtst, 100^2, :)
    xtest = xtst
    ytest = ytst
    DataLoader((xtest, ytest), batchsize=batch_size, shuffle=true)
end

function findrange(filename, variablename)
    xtrn, ytrn, xtst, ytst, highend, lowend  = loaddata(filename, variablename)
    return highend, lowend 
end

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim, sigmoid)
    #Dense(hidden_dim, input_dim)
)


function reconstuct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(z)
end

const F = Float32
function binary_cross_entropy(x, x̂)
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -s
    #return -mean(s)
end

function model_loss(encoder, decoder, decoder_params, λ, x, device)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    #logp_x_z = binary_cross_entropy(x, decoder_z)/len 
    #logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    logp_x_z = -binarycrossentropy(decoder_z, x, agg=sum) / len
    # regularization
    #reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
    reg = λ * sum(x->sum(x.^2), decoder_params)

    -logp_x_z + kl_q_p + reg
end


#### images plot
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 100, :)...), (2, 1)))
end

#Arguments for the `train` function
@with_kw mutable struct Args
    η = 1e-4                # learning rate
    λ = 0.0001f0              # regularization paramater
    batch_size = 100        # batch size
    sample_size = 4        # sampling size for output
    #epochs = 1             # number of epochs
    #seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 100^2        # image size
    #latent_dim = 100          # latent dimension
    #hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function train(filename, variablename, epochs, seed, latent_dim, hidden_dim; kws...)
    # load hyperparamters
    args = Args(; kws...)
    seed > 0 && Random.seed!(seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load train data
    loader = get_data(args.batch_size,filename,variablename)
    loader2 = get_data2(args.batch_size,filename,variablename)

    # initialize encoder and decoder
    encoder = Encoder(args.input_dim, latent_dim, hidden_dim) |> device
    decoder = Decoder(args.input_dim, latent_dim, hidden_dim) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
   
    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    #!ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger 
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end
    
    # input plot
    original, _ = first(get_data(args.sample_size,filename,variablename))
    original = original |> device
    image = convert_to_image(original, args.sample_size)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)
    @info "Image saved: $(image_path)"

    original2, _ = first(get_data2(args.sample_size,filename,variablename))
    original2 = original2 |> device
    image = convert_to_image(original2, args.sample_size)
    image_path = joinpath(args.save_path, "original2.png")
    save(image_path, image)
    @info "Image saved: $(image_path)"
    
    # training
    train_steps = 0
    @info "Start Training, total $(epochs) epochs"
    decoder_params = Flux.params(decoder)
    for epoch = 1:epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x,_) in loader
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, decoder_params, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end

            train_steps += 1
        end

    end
    # save model
    highend, lowend = findrange(filename, variablename)

    model_path = joinpath(args.save_path, "model.bson")
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        BSON.@save model_path encoder decoder args #highend lowend 
        @info "Model saved: $(model_path)"
    end
    _, _, rec_original = reconstuct(encoder, decoder, original, device)
    #rec_original = sigmoid.(rec_original)
    image = convert_to_image(rec_original, args.sample_size)
    image_path = joinpath(args.save_path, "output.png")
    save(image_path, image)
    @info "Image saved: $(image_path)"
    
    _, _, rec_original2 = reconstuct(encoder, decoder, original2, device)
    #rec_original2 = sigmoid.(rec_original2)
    image = convert_to_image(rec_original2, args.sample_size)
    image_path = joinpath(args.save_path, "output2.png")
    save(image_path, image)
    @info "Image saved: $(image_path)"
     
    loader3 = get_data2(1,filename,variablename) ## 1 is the batch size
    #device = cpu
    #BSON.@load "/lcldata/wuhao/RegAE_flux/RegAE_nz/output/model.bson" encoder decoder
    encoder = cpu(encoder)
    decoder = cpu(decoder)
    tstM = ones(length(loader3), latent_dim)
    i = 0
    for (x,_) in loader3
        μ, logσ = encoder(x)
        i = i + 1
        tstM[i,:] = μ
    end

    sigma = Statistics.cov(tstM; dims=1)
    mu = vec(Statistics.mean(tstM; dims=1))

    return encoder, decoder, highend, lowend, sigma, mu
end

#if abspath(PROGRAM_FILE) == @__FILE__
#    train()
#end

#train(100,1,100,500)

 
