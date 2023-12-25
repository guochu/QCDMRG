using JSON


function read_data(file)
	data = JSON.parsefile(file)

	println(data["times"])

	println(data["energies"])
end