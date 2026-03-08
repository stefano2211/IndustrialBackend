output "instance_public_ip" {
  description = "Persistent Public IP address (Elastic IP) of the EC2 instance"
  value       = aws_eip.app_eip.public_ip
}

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.app_server.id
}
