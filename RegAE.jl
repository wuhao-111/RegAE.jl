module RegAE

using FileIO, Flux, ArgParse, Images, Random, Statistics
import Distributed
import JLD2
import Optim
import Zygote
import NNlib
using NNlib
using BSON

include("vae.jl")

mutable struct Autoencoder{T, S, F, M, D}
       encoder::T
       decoder::S
       highend::F
       lowend::F
       sigma::M
       mu::D
end

function Autoencoder(vaefilename::String, vaefilename2::String)
 	#BSON.@load vaefilename encoder decoder
	encoder = BSON.load(vaefilename, @__MODULE__)[:encoder]
	decoder = BSON.load(vaefilename, @__MODULE__)[:decoder]
        @JLD2.load vaefilename2 highend lowend sigma mu 
        return Autoencoder(encoder, decoder, highend, lowend, sigma, mu)
end

function Autoencoder(filename::String, variablename::String, vaefilename::String, vaefilename2::String, epochs, seed, latent_dim, hidden_dim)
        if !isfile(vaefilename)
                encoder, decoder, highend, lowend, sigma, mu = train(filename, variablename, epochs, seed, latent_dim, hidden_dim)
		encoder = cpu(encoder)
                decoder = cpu(decoder)
                BSON.@save vaefilename encoder decoder
                @JLD2.save vaefilename2 highend lowend sigma mu
                return Autoencoder(encoder, decoder, highend, lowend, sigma, mu)
        else
                return Autoencoder(vaefilename, vaefilename2)
        end
end

function p2z(ae::Autoencoder, p)
        μ, logσ = ae.encoder(reshape((p .- ae.lowend) ./ (ae.highend - ae.lowend), size(p)..., 1))
	z = μ 
        return z
end

function z2p(ae::Autoencoder, z)
        p_normalized = ae.decoder(z)
        #p_normalized = σ.(p_normalized)
        p = ae.lowend .+ p_normalized .* (ae.highend - ae.lowend)
        return p
end

function gradient(z, objfunc, h)
        zs = map(i->copy(z), 1:length(z) + 1)
        for i = 1:length(zs) - 1
                zs[i][i] += h
        end
        ofs = Distributed.pmap(objfunc, zs; batch_size=ceil(length(z) / Distributed.nworkers()))
        return (ofs[1:end - 1] .- ofs[end]) ./ h
end

function optimize(ae::Autoencoder, objfunc, options, nz; h=1e-4, p0=false)
        objfunc_z = z->sum(z .^ 2) + objfunc(z2p(ae, z))
        if p0 == false
                z0 = zeros(nz)
        else
                z0 = p2z(ae, p0)
        end
        opt = Optim.optimize(objfunc_z, z->gradient(z, objfunc_z, h), z0, Optim.LBFGS(), options; inplace=false)
        return z2p(ae, opt.minimizer), opt
end

function optimize_zygote(ae::Autoencoder, objfunc, options, nz; h=1e-4, p0=false)
	objfunc_z = z->sum((z - ae.mu).*(ae.sigma\(z-ae.mu))) + objfunc(z2p(ae, z))
        if p0 == false
                z0 = zeros(nz)
        else
                z0 = p2z(ae, p0)
        end
        opt = Optim.optimize(objfunc_z, z->Zygote.gradient(objfunc_z, z)[1], z0, Optim.LBFGS(), options; inplace=false)
        return z2p(ae, opt.minimizer), opt
end

end
