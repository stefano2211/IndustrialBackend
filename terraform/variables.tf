variable "aws_region" {
  description = "The AWS region to deploy the infrastructure in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "The EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "ssh_public_key" {
  description = "Public SSH key for the EC2 instance (Used by GitHub Actions to connect)"
  type        = string
}

variable "project_name" {
  description = "Name of the project to tag resources"
  type        = string
  default     = "aura-industrial"
}
