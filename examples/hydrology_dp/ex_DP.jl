using Distributed
using LaTeXStrings
import Flux
import Optim
import PyPlot
import StatsBase
import Zygote

@everywhere begin
	import DPFEHM
	import GeostatInversion
	import JLD2
	import Random
	import RegAE
	import SharedArrays
        import Statistics

	Random.seed!(myid())

	sidelength = 50.0#m
	thickness = 10.0#m
	mins = [-sidelength, -sidelength, 0]
	maxs = [sidelength, sidelength, thickness]
	ns = [100, 100, 1]
	meanloghyco1 = log(1e-5)#m/s
	meanloghyco2 = log(1e-8)#m/s
	lefthead = 1.0#m
	righthead = 0.0#m
 	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)

	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	zs = range(mins[3]; stop=maxs[3])
	function samplehyco!(fields::SharedArrays.SharedArray; setseed=false)
		if nworkers() == 1 || size(fields, 1) == 3#if it is small or there is only one processor break up the chunk more simply
			if myid() <= 2
				mychunk = 1:size(fields, 1)
			else
				mychunk = 1:0
			end
		else
			mychunk = 1 + div((myid() - 2) * size(fields, 1), nworkers()):div((myid() - 1) * size(fields, 1), nworkers())
		end
		for i in mychunk
			if setseed
				Random.seed!(i)
			end
			field1 = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco1, 0.5, -3.0)#fairly smooth field
			field2 = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco2, 0.5, -2.5)#fairly rough field
			fieldsplit = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco2, 1, -4.5)#very smooth field
			p = Random.rand()
			splitnumber = Statistics.quantile(fieldsplit[:], 0.25 + 0.5 * p)
			for j1 = 1:size(fields, 2)
				for j2 = 1:size(fields, 3)
					if fieldsplit[j1, j2] > splitnumber
						fields[i, j1, j2] = field2[j1, j2]
					else
						fields[i, j1, j2] = field1[j1, j2]
					end
				end
			end
		end
		return nothing
	end

	boundaryhead(x, y) = (lefthead-righthead) * (x - maxs[1]) / (mins[1] - maxs[1])
	dirichletnodes = Int[]
	dirichletheads = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
        	if coords[1, i] == mins[1] || coords[1, i] == maxs[1]
                	push!(dirichletnodes, i)
                	dirichletheads[i] = boundaryhead(coords[1:2, i]...)
        	end
	end

end

variablename = "allloghycos"
datafilename = "trainingdata.jld2"
if !isfile(datafilename)
	numsamples = 10^5
	@time allloghycos = SharedArrays.SharedArray{Float32}(numsamples, ns[2], ns[1]; init=A->samplehyco!(A; setseed=true))
	@time @JLD2.save datafilename allloghycos
end
@JLD2.load datafilename allloghycos

ae = RegAE.Autoencoder(datafilename, variablename, "vae_nz25.bson", "vae_nz25.jld2", 100, 1, 25, 125)
ae = RegAE.Autoencoder(datafilename, variablename, "vae_nz50.bson", "vae_nz50.jld2", 100, 1, 50, 250)
ae = RegAE.Autoencoder(datafilename, variablename, "vae_nz100.bson", "vae_nz100.jld2", 100, 1, 100, 500)
ae = RegAE.Autoencoder(datafilename, variablename, "vae_nz200.bson", "vae_nz200.jld2", 100, 1, 200, 1000)
ae = RegAE.Autoencoder(datafilename, variablename, "vae_nz400.bson", "vae_nz400.jld2", 100, 1, 400, 2000)
#Autoencoder(vaefilename::String, vaefilename2::String, epochs, seed, latent_dim, hidden_dim)

@everywhere Random.seed!(0)
p_trues = Array(SharedArrays.SharedArray{Float32}(3, ns[2], ns[1]; init=samplehyco!))
casenames = ["nz25","nz50", "nz100", "nz200", "nz400"]
nzs = [25, 50, 100, 200, 400]
for i_p in 1:size(p_trues, 1)
	for i_case in 1:length(casenames)
		if !isfile("opt_$(i_p)_$(i_case).jld2")
			casename = casenames[i_case]
			nz = nzs[i_case]
                        p_true = p_trues[i_p, :, :]
			ae = RegAE.Autoencoder("vae_$(casename).bson", "vae_$(casename).jld2")
			indices = reshape(collect(1:10^4), 100, 100)
			obsindices = indices[17:17:100, 17:17:100][:]
			p_indices = reshape(collect(1:10^4), 100, 100)
			p_obsindices = p_indices[17:17:100, 17:17:100][:]
			logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
			sources = zeros(size(coords, 2))
			function gethead(p)
				loghycos = p
				if maximum(loghycos) - minimum(loghycos) > 25
                                	return fill(NaN, length(sources))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
				else
					neighborhycos = logKs2Ks_neighbors(loghycos)
					#@show extrema(neighborhycos)
					head = DPFEHM.groundwater_steadystate(neighborhycos, neighbors, areasoverlengths, dirichletnodes, dirichletheads, sources; reltol=1e-12)
					#@show extrema(head)
					if maximum(head) > lefthead + 1e-4 || minimum(head) < righthead - 1e-4
						error("problem with solution -- head out of range")
					end
					return head
				end
			end
			head_true = reshape(gethead(p_true), size(p_true)...)
			function objfunc(p_flat; show_parts=false)
				p = reshape(p_flat, size(p_true)...)
				head = reshape(gethead(p), size(head_true)...)
				if show_parts
					@show sqrt(sum((head[obsindices] .- head_true[obsindices]) .^ 2) / length(obsindices))
					@show sqrt(sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2) / length(p_obsindices))
				end
				return 1e4 * sum((head[obsindices] .- head_true[obsindices]) .^ 2) + 3e0 * sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2)
			end
			@time p_flat, opt = RegAE.optimize_zygote(ae, objfunc, Optim.Options(iterations=200, extended_trace=false, store_trace=true, show_trace=false), nz)
			@JLD2.save "opt_$(i_p)_$(i_case).jld2" p_true p_flat opt
			fig, axs = PyPlot.subplots(1, 2)
			axs[1].imshow(reshape(RegAE.z2p(ae, opt.minimizer), size(p_true)...), cmap="jet", vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			p = RegAE.z2p(ae, opt.minimizer)
			@show minimum(p), maximum(p) 
			axs[2].imshow(p_trues[i_p,: ,:], cmap="jet", vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			#display(fig)
			#println()
  			fig.savefig("ex3_result_$(i_case)_$(i_p).pdf")
  			PyPlot.close(fig)
  		end
  	end
end

##### the part for p2zz2p plot
#for i_p in 1:size(p_trues, 1)
#	for i_case in 1:length(casenames)
#		casename = casenames[i_case]
#		ae = RegAE.Autoencoder("vae_$(casename).bson", "vae_$(casename).jld2")
#		p_true = p_trues[i_p, :, :]
#        	p_vae = RegAE.z2p(ae, RegAE.p2z(ae,reshape(p_true, 100^2)))
#		#println("nz$(i_case)_fig$(i_p)")
#		#println(minimum(p_vae))
#		#println(maximum(p_vae))
#        	fig,axs = PyPlot.subplots(1,2,figsize=(16,9))
#		axs[1].imshow(p_true, cmap="jet", vmin = ae.lowend,vmax = ae.highend, extent=[-sidelength,sidelength,-sidelength,sidelength],origin="lower",interpolation="nearest")
#		axs[1].title.set_text("true")
#		axs[2].imshow(reshape(p_vae, size(p_true)...), cmap="jet", vmin = ae.lowend,vmax = ae.highend, extent=[-sidelength,sidelength,-sidelength,sidelength],origin="lower",interpolation="nearest")
#		axs[2].title.set_text("prediction")
#		fig.savefig("pz_$(i_case)_$(i_p).pdf")
#		PyPlot.close(fig)
#	end
#end
#### p2zz2p plot finished

### the part for comparison plot 
#indices = reshape(collect(1:10^4), 100, 100)
#obsindices = indices[17:17:100, 17:17:100][:]
#p_indices = reshape(collect(1:10^4), 100, 100)
#p_obsindices = p_indices[17:17:100, 17:17:100][:]
#logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
#sources = zeros(size(coords, 2))
#function gethead(p)
#	loghycos = p
#	if maximum(loghycos) - minimum(loghycos) > 25
#        	return fill(NaN, length(sources))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
#	else
#		neighborhycos = logKs2Ks_neighbors(loghycos)
#		head = DPFEHM.groundwater_steadystate(neighborhycos, neighbors, areasoverlengths, dirichletnodes, dirichletheads, sources; reltol=1e-12)
#		if maximum(head) > lefthead + 1e-4 || minimum(head) < righthead - 1e-4
#			error("problem with solution -- head out of range")
#		end
#		return head
#	end
#end

#for i_p in 1:size(p_trues, 1)
#	for i_case in 1:length(casenames)
#		casename = casenames[i_case]
#		ae = RegAE.Autoencoder("vae_$(casename).bson", "vae_$(casename).jld2")
#        	@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_true
#        	@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_flat
#		fig, axs = PyPlot.subplots(1,2, figsize=(12,6))
#		head = gethead(p_flat)
#	    	head_true = gethead(p_true)
#		axs[1].plot(head[obsindices], head_true[obsindices], ".")	
#		axs[1].plot([0, 1], [0, 1], "r", alpha=0.5)
#		axs[1].title.set_text("head")
#		axs[1].set_xlabel("prediction")
#		axs[1].set_ylabel("true")
#		axs[2].plot(p_flat[p_obsindices], p_true[p_obsindices], ".")
#		axs[2].plot([ae.lowend, ae.highend], [ae.lowend, ae.highend], "r", alpha=0.5)
#                axs[2].title.set_text("permeability")
#		axs[2].set_xlabel("prediction")
#		axs[2].set_ylabel("true")
#		fig.savefig("comp_$(i_case)_$(i_p).pdf")
#		PyPlot.close(fig)
#	end
#end
###comparison plot finished

representations =[L"n_z=25",L"n_z=50", L"n_z=100", L"n_z=200", L"n_z=400"]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"] 
fig, axs = PyPlot.subplots(7, 3; figsize = (12, 30)) 
k = 1 
for i_p in 1:size(p_trues, 1) 
	global k 
        @JLD2.load "opt_$(i_p)_1.jld2" p_true
        axs[1, i_p].imshow(p_trues[i_p, :, :], vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet") 
	axs[1, i_p].set_aspect("equal") 
	axs[1, i_p].set_title("Reference Field") 
        axs[1, i_p].set_ylabel(L"y") 
        axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))") 
        k += 1 
        for i_case in 1:length(casenames) 
        	@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_flat 
        	p_opt = reshape(p_flat, size(p_trues[1, : ,:]) ...) 
		axs[i_case + 1, i_p].imshow(p_opt, vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
    		axs[i_case + 1, i_p].set_aspect("equal")
    		axs[i_case + 1, i_p].set_title(representations[i_case] * ", Relative Error: $(round(sum((p_opt .- p_trues[i_p, :, :]) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=2))")
    		axs[i_case + 1, i_p].set_ylabel(L"y") 
        	axs[i_case + 1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
    		k += 1 
        	@JLD2.load "opt_$(i_p)_$(i_case).jld2" opt 
		axs[7, i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), label=representations[i_case], lw=3, alpha =0.5) 
	end
    	axs[7, i_p].legend()
	axs[7, i_p].set_ylabel("Objective Function") 
	axs[7, i_p].set_xlabel("Iteration\n($(alphabet[k]))") 
	k += 1 
end 
fig.tight_layout()
display(fig) 
fig.savefig("megaplot.pdf") 
println()
PyPlot.close(fig) 


fig, axs = PyPlot.subplots(2, 3; figsize = (12, 8)) 
k = 1 
for i_p in 1:size(p_trues, 1) 
	global k 
  	@JLD2.load "opt_$(i_p)_1.jld2" p_true
        axs[1, i_p].imshow(p_trues[i_p, :, :], vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet") 
	axs[1, i_p].set_aspect("equal") 
	axs[1, i_p].set_title("Reference Field") 
	axs[1, i_p].set_ylabel(L"y") 
	axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))") 
	k += 1 
	for i_case in 1:length(casenames) 
		if i_case == 2 
			@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_flat 
			p_opt = reshape(p_flat, size(p_trues[1,: ,:]) ...) 
			axs[i_case, i_p].imshow(p_opt, vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
      			axs[i_case, i_p].set_aspect("equal")
      			axs[i_case, i_p].set_title(representations[i_case] * ", Relative Error: $(round(sum((p_opt .- p_trues[i_p, :, :]) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=2))")
      			axs[i_case, i_p].set_ylabel(L"y")
			axs[i_case, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
			k += 1
		end
	end
end
fig.tight_layout()
display(fig)
fig.savefig("comparison.pdf")
println()
PyPlot.close(fig)


fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
	for i_case in 1:length(casenames)
		@JLD2.load "opt_$(i_p)_$(i_case).jld2" opt
		axs[i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), label=representations[i_case], lw=3, alpha=0.5)
	end
	axs[i_p].legend()
	axs[i_p].set_ylabel("Objective Function")
	axs[i_p].set_xlabel("Iteration\n($(alphabet[k]))")
	k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("convergence.pdf")
println()
PyPlot.close(fig)
