include("/home/trung/_qhe-julia/FQH_state_v2.jl")
include("/home/trung/_qhe-julia/Potentials.jl")
include("/home/trung/_qhe-julia/Misc.jl")
#include("/home/trung/two_body_potential/v1_sphere.jl")
include("/home/trung/_qhe-julia/HilbertSpace.jl")

using .FQH_states
using .Potentials
using .MiscRoutine
#using .PseudoPotential
using Main.HilbertSpaceGenerator
using LinearAlgebra
using SparseArrays
using ArgMacros

CONFIG_FILE_PATH = "CHS.config"
AVAILABLE_PIN_TYPES = ["4pins","4pins_positive","4pins_positive_displaced",
                "north","positive_north","positive_samesign","same_sign"]

function isadmissible(partition::BitVector, k::Integer, r::Integer)
    check = true
    for i in 1:(length(partition)-r+1)
        if count(partition[i:(i+r-1)]) > k
            check=false
            break
        end
    end
    return check
end

function fileconfigure(filename,ask_confirm=true)
    if isfile(filename)
        if ask_confirm
            println("File '$(filename)' exists.")
            println("Configuring the script will permanently modify the code. Continue? (Y/N)")
            doit = lowercase(readline()) in ["y","yes"]
        else
            doit = true
        end
        if doit
            script_path = @__FILE__
            code = read(script_path,String)

            new_code = replace(code, "CONFIG_FILE_PATH = \"$(CONFIG_FILE_PATH)\"" => "CONFIG_FILE_PATH = \"$(filename)\"")
            open(script_path,"w+") do f
                write(f,new_code)
            end
            println("The script was configured successfully.")
        else
            println("The script was NOT configured.")
        end
    else
        println("File '$(filename)' not found.")
        println("The script was NOT configure")
    end
end

function main()
    @inlinearguments begin
        @argumentoptional String fname "-f" "--filename"
        @argumentoptional String bname "-b" "--basisname"
        @argumentoptional String rname "-r" "--rootname"
        @argumentoptional Int Ne "--n_el" "-e"
        @argumentoptional Int No "--n_orb" "-o"
        @argumentoptional Int Lz "--L_z"
        @argumentoptional String appendfile "--append"
        @argumentoptional Float64 appendparam "--append-param"
        @argumentoptional String fqhphase "--phase"
        @argumentflag decimal_file "--decimal"
        @argumentflag configure "--configure"
        @argumentflag configure_default "--configure-default"
    end

    # Configure only
    if configure
        println("Specify the configuration file name below. (The default name is 'CHS.config'.)")
        fileconfigure(readline())
        return
    end

    if configure_default
        fileconfigure("CHS.config",false)
        return
    end

    # Read wavefunction
    if bname == nothing
        if decimal_file
            state = readwfdec(fname,No)
        else
            state = readwf(fname)
        end
    else
        state = readwf(bname,fname,No)
    end
    println("Check norm = $(wfnorm(state))")

    # Check valid phase
    if !(fqhphase in keys(JACKS_DIR))
        println("Argument not recognized. Available keys are:")
        for p in keys(JACKS_DIR) print("$p\t") end
        println("\nTerminating.")
        return
    else
        println("Taking cumulative overlap in the $(fqhphase) CHS.")
    end

    # Get list of root configurations
    if rname == nothing
        if Lz == nothing
            basis = fullhilbertspace(Ne,No)
        else
            basis = fullhilbertspace(Ne,No,Lz)
        end
    else
        println("Reading roots from file $rname")
        basis = readwf(rname).basis
    end 

    # Get admissible roots
    if fqhphase == "Laughlin"
        roots = basis[map(x->isadmissible(x,1,3), basis)]
    elseif fqhphase == "Gaffnian"
        roots = basis[map(x->isadmissible(x,2,5), basis)]
    elseif fqhphase == "Moore-Read"
        roots = basis[map(x->isadmissible(x,2,4), basis)]
    end
    println("$(length(roots)) admissible root(s).")
    println("------")

    # Prepare to read the jacks
    jack_list = Vector{FQH_state}()

    S = (No-1.)/2

    println("\n\n$(Ne) electrons and $(No) orbitals\n")

    println("S = $S on the sphere.")

    # Read all the jacks
    Lz_sectors = Dict{Float64,Vector{T} where T<:AbstractFQH_state}()
    for root in roots
        L = findLzsphere(root,S)
        if L == Lz || Lz == nothing
            rootstring = prod(string.((Int.(root))))
            print("\r$(rootstring)\t")
            try
                jack = sphere_normalize(readwf("$(JACKS_DIR[fqhphase])/J_$(rootstring)";verbose=false))
                if haskey(Lz_sectors,L)
                    push!(Lz_sectors[L], jack)
                else
                    Lz_sectors[L] = [jack]
                end
            catch LoadError
                continue
            end
        end
    end

    println("Lz sectors = ")
    println(sort(collect(keys(Lz_sectors))))

    # For each L_z, orthonormalize and add to the cumulative overlap
    overlap_by_Lz = Float64[]
    total_overlap = 0.
    if Lz != nothing
        @time begin
        println("Working on Lz = $(Lz) sector")
        all_basis, all_coef = collate_many_vectors(Lz_sectors[Lz]; separate_out=true, collumn_vector=true)

        println("Orthonormalizing basis using QR decomposition")
        @time all_coef_ortho = transpose(Matrix(qr(all_coef).Q))

        dim = length(all_basis)
        d   = size(all_coef_ortho)[1]

        println("$(d) state(s) at L_z = $(Lz) sector. $(dim) monomials.")
       
        state_coef = projection_coefficients(state,all_basis)
        println("Norm of projected state: $(sqrt(sum(abs2.(state_coef))))")
        total_overlap += sum(abs2.(all_coef_ortho * state_coef))

        end
        println("------")

        open("$(fname)_ov_$(fqhphase)_Lz_$(Lz).txt","w+") do f
            write(f,"$(total_overlap)\n")
        end

    else
        for L_z in sort(collect(keys(Lz_sectors)))
            # Orthogonalize all jacks
            @time begin
            println("Working on Lz = $(L_z) sector")
            all_basis, all_coef = collate_many_vectors(Lz_sectors[L_z]; separate_out=true, collumn_vector=true)

            println("Orthonormalizing basis using QR decomposition")
            @time all_coef_ortho = transpose(Matrix(qr(all_coef).Q))

            dim = length(all_basis)
            d   = size(all_coef_ortho)[1]

            println("$(d) state(s) at L_z = $(L_z) sector. $(dim) monomials.")
           
            state_coef = projection_coefficients(state,all_basis)
            println("Norm of projected state: $(sqrt(sum(abs2.(state_coef))))")
            overlap = sum(abs2.(all_coef_ortho * state_coef))
            #overlap = sum(i -> abs2(state ⋅ FQH_state(all_basis,all_coef_ortho[i,:])), 1:d)
            println("Cumulative overlap with Lz = $(L_z) sector: $(overlap)")
            push!(overlap_by_Lz,overlap)
            total_overlap += overlap

            end # end @time
            println("------")
        end

        open("$(fname)_ov_$(fqhphase).txt","w+") do f
            write(f,"$(total_overlap)\n")
            for (x,y) in zip(sort(collect(keys(Lz_sectors))), overlap_by_Lz)
                write(f,"$x\t$y\n")
            end
        end
    end

    println("===========")
    println("TOTAL OVERLAP = $(total_overlap)")

    if appendfile != nothing
        open(appendfile,"a+") do f
            if appendparam == nothing
                write(f,"$(total_overlap)")
            else
                write(f,"$(appendparam)\t$(total_overlap)\n")
            end
        end
    end
end


if isfile(CONFIG_FILE_PATH)
        include(CONFIG_FILE_PATH)
        @time main()
    else
        println("Config file not found, so the locations where the jacks are stored cannot be found.")
        println("Terminating.")
    end